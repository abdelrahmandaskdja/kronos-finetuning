import os
import sys
import json
import time
import pickle
import random
import subprocess
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append('../')
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig
from training_charts import save_training_dashboard


PRICE_CLOSE_COL = 3
_BIT_TABLE_CACHE = {}


def _get_bit_table(bits: int, device: torch.device) -> torch.Tensor:
    cache_key = (bits, str(device))
    if cache_key in _BIT_TABLE_CACHE:
        return _BIT_TABLE_CACHE[cache_key]

    token_ids = torch.arange(2 ** bits, dtype=torch.long, device=device)
    bit_mask = 2 ** torch.arange(bits, dtype=torch.long, device=device)
    bit_table = ((token_ids.unsqueeze(-1) & bit_mask) != 0).float() * 2 - 1
    _BIT_TABLE_CACHE[cache_key] = bit_table
    return bit_table


def _decode_soft_predictions(tokenizer: KronosTokenizer, s1_logits: torch.Tensor, s2_logits: torch.Tensor) -> torch.Tensor:
    probs_s1 = torch.softmax(s1_logits, dim=-1)
    probs_s2 = torch.softmax(s2_logits, dim=-1)

    bit_table_s1 = _get_bit_table(tokenizer.s1_bits, probs_s1.device)
    bit_table_s2 = _get_bit_table(tokenizer.s2_bits, probs_s2.device)

    soft_bits_s1 = torch.matmul(probs_s1, bit_table_s1)
    soft_bits_s2 = torch.matmul(probs_s2, bit_table_s2)
    soft_bits = torch.cat([soft_bits_s1, soft_bits_s2], dim=-1)

    q_scale = 1.0 / (tokenizer.codebook_dim ** 0.5)
    soft_bits = soft_bits * q_scale

    z = tokenizer.post_quant_embed(soft_bits)
    for layer in tokenizer.decoder:
        z = layer(z)
    z = tokenizer.head(z)
    return z


def _directional_loss_and_accuracy(pred_seq: torch.Tensor, target_seq: torch.Tensor):
    pred_close = pred_seq[:, :, PRICE_CLOSE_COL]
    target_close = target_seq[:, :, PRICE_CLOSE_COL]

    pred_delta = pred_close[:, 1:] - pred_close[:, :-1]
    target_delta = target_close[:, 1:] - target_close[:, :-1]

    target_up = (target_delta >= 0).float()
    dir_loss = F.binary_cross_entropy_with_logits(pred_delta, target_up)
    dir_acc = ((pred_delta >= 0) == (target_up > 0.5)).float().mean()
    return dir_loss, dir_acc


def _relative_move_magnitude(
    current_close: torch.Tensor,
    prev_close: torch.Tensor,
    *,
    scale: float = 0.01,
    cap: float = 5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    prev_close_safe = prev_close.abs().clamp_min(float(eps))
    rel_move = (current_close - prev_close).abs() / prev_close_safe
    scale = max(float(scale), float(eps))
    scaled_move = rel_move / scale
    if cap > 0.0:
        scaled_move = scaled_move.clamp(max=float(cap))
    return rel_move, scaled_move


def _context_window_move_magnitude(
    current_close: torch.Tensor,
    prev_close: torch.Tensor,
    context_close_seq: torch.Tensor,
    *,
    cap: float = 5.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    abs_move = (current_close - prev_close).abs()
    if context_close_seq.ndim != 2:
        raise ValueError(
            "context_close_seq must have shape (batch, context_len) for context-window magnitude weighting"
        )

    if context_close_seq.size(1) >= 2:
        context_deltas = (context_close_seq[:, 1:] - context_close_seq[:, :-1]).abs()
        context_scale = context_deltas.mean(dim=1, keepdim=True)
    else:
        # Degenerate fallback keeps weighting finite when the configured context length is tiny.
        context_scale = abs_move.mean(dim=1, keepdim=True)

    context_scale = context_scale.clamp_min(float(eps))
    scaled_move = abs_move / context_scale
    if cap > 0.0:
        scaled_move = scaled_move.clamp(max=float(cap))
    return abs_move, scaled_move


def _consecutive_directional_loss_and_accuracy(
    pred_seq: torch.Tensor,
    target_seq: torch.Tensor,
    context_close: torch.Tensor,
    context_close_seq: torch.Tensor | None = None,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    negative_weight: float = 1.0,
    false_positive_weight: float = 0.0,
    bias_penalty_weight: float = 0.0,
    magnitude_weight: float = 0.0,
    magnitude_mode: str = "relative_return",
    magnitude_scale: float = 0.01,
    magnitude_power: float = 1.0,
    magnitude_cap: float = 5.0,
):
    """Directional objective aligned with consecutive rollout metric.

    - Prediction direction compares close[t] vs previous predicted close.
    - Target direction compares close[t] vs previous actual close.
    - First step uses the last context close for both references.
    """
    pred_close = pred_seq[:, :, PRICE_CLOSE_COL]
    target_close = target_seq[:, :, PRICE_CLOSE_COL]

    context_close = context_close.to(pred_close.dtype).unsqueeze(1)
    prev_actual_close = torch.cat([context_close, target_close[:, :-1]], dim=1)
    prev_pred_close = torch.cat([context_close, pred_close[:, :-1]], dim=1)

    pred_delta = pred_close - prev_pred_close
    target_up_hard = (target_close >= prev_actual_close).float()
    target_up = target_up_hard

    if label_smoothing > 0.0:
        smoothing = float(label_smoothing)
        target_up = target_up * (1.0 - 2.0 * smoothing) + smoothing

    bce = F.binary_cross_entropy_with_logits(
        pred_delta,
        target_up,
        reduction="none",
        pos_weight=pred_delta.new_tensor(float(pos_weight)),
    )
    loss_terms = bce
    if focal_gamma > 0.0:
        p = torch.sigmoid(pred_delta)
        pt = p * target_up + (1.0 - p) * (1.0 - target_up)
        focal_factor = (1.0 - pt).clamp_min(1e-6).pow(float(focal_gamma))
        loss_terms = focal_factor * loss_terms
    else:
        p = torch.sigmoid(pred_delta)

    negative_weight = max(float(negative_weight), 0.0)
    if abs(negative_weight - 1.0) > 1e-12:
        down_mask = 1.0 - target_up_hard
        class_factor = target_up_hard + negative_weight * down_mask
        loss_terms = loss_terms * class_factor

    false_positive_weight = max(float(false_positive_weight), 0.0)
    if false_positive_weight > 0.0:
        down_mask = 1.0 - target_up_hard
        false_positive_factor = 1.0 + false_positive_weight * down_mask * p
        loss_terms = loss_terms * false_positive_factor

    if magnitude_weight > 0.0:
        magnitude_mode = str(magnitude_mode).strip().lower()
        if magnitude_mode in {"relative_return", "return"}:
            _, target_move_scaled = _relative_move_magnitude(
                current_close=target_close,
                prev_close=prev_actual_close,
                scale=magnitude_scale,
                cap=magnitude_cap,
            )
        elif magnitude_mode in {"context_window", "context_abs_delta", "context_mean_abs_delta"}:
            if context_close_seq is None:
                raise ValueError(
                    "context_window magnitude mode requires context_close_seq to calibrate big vs small moves"
                )
            _, target_move_scaled = _context_window_move_magnitude(
                current_close=target_close,
                prev_close=prev_actual_close,
                context_close_seq=context_close_seq.to(target_close.dtype),
                cap=magnitude_cap,
            )
        else:
            raise ValueError(
                f"Unsupported consecutive directional magnitude mode: {magnitude_mode}. "
                "Use 'relative_return' or 'context_window'."
            )
        move_weight = 1.0 + float(magnitude_weight) * target_move_scaled.pow(float(magnitude_power))
        loss_terms = loss_terms * move_weight

    dir_loss = loss_terms.mean()
    bias_penalty_weight = max(float(bias_penalty_weight), 0.0)
    if bias_penalty_weight > 0.0:
        target_up_ratio = target_up_hard.mean(dim=1)
        pred_up_ratio = p.mean(dim=1)
        bias_penalty = F.mse_loss(pred_up_ratio, target_up_ratio)
        dir_loss = dir_loss + bias_penalty_weight * bias_penalty

    dir_acc = ((pred_delta >= 0) == (target_up_hard > 0.5)).float().mean()
    return dir_loss, dir_acc


def _scheduled_sampling_rate(config, epoch_idx: int) -> float:
    max_rate = float(getattr(config, 'predictor_scheduled_sampling_max_rate', 0.0))
    max_rate = float(np.clip(max_rate, 0.0, 1.0))
    if max_rate <= 0.0:
        return 0.0

    warmup_epochs = max(int(getattr(config, 'predictor_scheduled_sampling_warmup_epochs', 0)), 0)
    ramp_epochs = max(int(getattr(config, 'predictor_scheduled_sampling_ramp_epochs', 0)), 0)

    if epoch_idx < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_rate

    progress = min(1.0, float(epoch_idx - warmup_epochs + 1) / float(ramp_epochs))
    return max_rate * progress


def _sample_token_ids(logits: torch.Tensor, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == 'sample':
        flat_logits = logits.reshape(-1, logits.size(-1))
        samples = torch.distributions.Categorical(logits=flat_logits).sample()
        return samples.reshape(logits.shape[:-1])
    return torch.argmax(logits, dim=-1)


def _apply_scheduled_sampling(
    model,
    use_ddp: bool,
    token_in: list[torch.Tensor],
    stamp: torch.Tensor,
    config,
    epoch_idx: int,
):
    strategy = str(getattr(config, 'predictor_scheduled_sampling_strategy', 'tail')).strip().lower()
    if strategy in {'', 'none', 'off', 'disabled'}:
        return token_in, 0.0, 0.0

    rate = _scheduled_sampling_rate(config, epoch_idx)
    if rate <= 0.0:
        return token_in, 0.0, 0.0

    seq_len = int(token_in[0].shape[1])
    future_start = min(max(int(getattr(config, 'lookback_window', 0)), 1), seq_len)
    future_len = seq_len - future_start
    if future_len <= 0:
        return token_in, rate, 0.0

    wrapped_model = model
    base_model = model.module if use_ddp else model
    was_training = wrapped_model.training
    if was_training:
        wrapped_model.eval()

    with torch.no_grad():
        s1_logits, context = base_model.decode_s1(token_in[0], token_in[1], stamp)
        pred_s1 = _sample_token_ids(
            s1_logits,
            mode=str(getattr(config, 'predictor_scheduled_sampling_mode', 'greedy')),
        )
        s2_logits = base_model.decode_s2(context, pred_s1)
        pred_s2 = _sample_token_ids(
            s2_logits,
            mode=str(getattr(config, 'predictor_scheduled_sampling_mode', 'greedy')),
        )

    if was_training:
        wrapped_model.train()

    mixed_s1 = token_in[0].clone()
    mixed_s2 = token_in[1].clone()
    replace_ratio = 0.0

    if strategy == 'tail':
        tail_len = int(round(rate * future_len))
        if tail_len <= 0:
            return token_in, rate, 0.0
        replace_start = seq_len - tail_len
        mixed_s1[:, replace_start:] = pred_s1[:, replace_start - 1 : seq_len - 1]
        mixed_s2[:, replace_start:] = pred_s2[:, replace_start - 1 : seq_len - 1]
        replace_ratio = float(tail_len) / float(future_len)
    elif strategy == 'bernoulli':
        mask = torch.rand(
            mixed_s1.size(0),
            future_len,
            device=mixed_s1.device,
            dtype=torch.float32,
        ) < rate
        pred_slice_s1 = pred_s1[:, future_start - 1 : seq_len - 1]
        pred_slice_s2 = pred_s2[:, future_start - 1 : seq_len - 1]
        mixed_s1[:, future_start:] = torch.where(mask, pred_slice_s1, mixed_s1[:, future_start:])
        mixed_s2[:, future_start:] = torch.where(mask, pred_slice_s2, mixed_s2[:, future_start:])
        replace_ratio = float(mask.float().mean().item())
    else:
        raise ValueError(
            f"Unsupported predictor_scheduled_sampling_strategy={strategy}. "
            "Use 'tail', 'bernoulli', or 'none'."
        )

    return [mixed_s1, mixed_s2], rate, replace_ratio


def _normalize_checkpoint_metric(raw_metric: str) -> str:
    metric = str(raw_metric).strip().lower()
    aliases = {
        'loss': 'val_loss',
        'val_cons_dir_acc_pct': 'val_cons_dir_acc',
        'cons_dir_acc': 'val_cons_dir_acc',
        'rollout': 'rollout_metric',
    }
    return aliases.get(metric, metric)


def _metric_improved(
    current_value: float | None,
    best_value: float,
    *,
    higher_is_better: bool,
    min_delta: float = 0.0,
) -> bool:
    if current_value is None:
        return False

    min_delta = max(float(min_delta), 0.0)
    if higher_is_better:
        return float(current_value) > float(best_value) + min_delta
    return float(current_value) < float(best_value) - min_delta


class ModelEma:
    """CPU-backed EMA of model weights for cheaper steady-state evaluation."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(np.clip(decay, 0.0, 0.999999))
        self.shadow_state: dict[str, torch.Tensor] | None = None
        self.backup_state: dict[str, torch.Tensor] | None = None
        if self.decay > 0.0:
            self.shadow_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    @property
    def enabled(self) -> bool:
        return self.shadow_state is not None

    def has_state(self) -> bool:
        return self.shadow_state is not None

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.shadow_state is None:
            return
        for name, tensor in model.state_dict().items():
            shadow_tensor = self.shadow_state[name]
            current_cpu = tensor.detach().cpu()
            if torch.is_floating_point(current_cpu):
                shadow_tensor.mul_(self.decay).add_(current_cpu, alpha=1.0 - self.decay)
            else:
                shadow_tensor.copy_(current_cpu)

    @torch.no_grad()
    def store(self, model: nn.Module):
        self.backup_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        if self.shadow_state is None:
            return
        target_state = model.state_dict()
        for name, tensor in target_state.items():
            tensor.copy_(self.shadow_state[name].to(device=tensor.device, dtype=tensor.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup_state is None:
            return
        target_state = model.state_dict()
        for name, tensor in target_state.items():
            tensor.copy_(self.backup_state[name].to(device=tensor.device, dtype=tensor.dtype))
        self.backup_state = None


def _run_rollout_checkpoint_eval(model, tokenizer, device, config, epoch_idx: int, logger):
    train_context_path = str(getattr(config, 'predictor_rollout_train_context_path', '')).strip()
    eval_path = str(getattr(config, 'predictor_rollout_eval_path', '')).strip()
    if not train_context_path or not eval_path:
        raise ValueError(
            "predictor_checkpoint_metric=rollout_metric requires "
            "predictor_rollout_train_context_path and predictor_rollout_eval_path"
        )

    metric_key = str(
        getattr(
            config,
            'predictor_rollout_metric_key',
            'directional_accuracy_close_vs_prev_close_consecutive_path',
        )
    ).strip()
    lookback = int(getattr(config, 'predictor_rollout_lookback', 0) or getattr(config, 'lookback_window', 0))
    block_len = int(getattr(config, 'predictor_rollout_block_len', 0) or getattr(config, 'predict_window', 0))
    feedback_source = str(getattr(config, 'predictor_rollout_feedback_source', 'actual')).strip() or 'actual'
    rollout_device = str(getattr(config, 'predictor_rollout_device', '')).strip()
    if not rollout_device:
        rollout_device = str(device)
    trade_fee_bps = float(getattr(config, 'predictor_rollout_trade_fee_bps', 0.0))
    stabilize_output = bool(getattr(config, 'predictor_rollout_stabilize_output', True))
    stability_apply_to_context = bool(
        getattr(config, 'predictor_rollout_stability_apply_to_context', False)
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eval_script = os.path.join(os.path.dirname(__file__), 'eval_consecutive_rollout.py')
    out_dir = os.path.join(config.metrics_dir, 'rollout_eval', f'epoch_{epoch_idx + 1:03d}')
    model_dir = os.path.join(config.base_save_path, '_rollout_checkpoint_tmp')
    out_csv = os.path.join(out_dir, 'predictions.csv')
    out_json = os.path.join(out_dir, 'metrics.json')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)

    tokenizer_id = getattr(config, 'finetuned_tokenizer_path', '') or getattr(config, 'pretrained_tokenizer_path', '')
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{repo_root}:{env['PYTHONPATH']}" if env.get('PYTHONPATH') else repo_root

    cmd = [
        sys.executable,
        eval_script,
        '--train-context-path',
        train_context_path,
        '--eval-path',
        eval_path,
        '--model-id',
        model_dir,
        '--tokenizer-id',
        tokenizer_id,
        '--lookback',
        str(lookback),
        '--block-len',
        str(block_len),
        '--feedback-source',
        feedback_source,
        '--device',
        rollout_device,
        '--trade-fee-bps',
        str(trade_fee_bps),
        '--output-csv',
        out_csv,
        '--output-json',
        out_json,
    ]
    if not stabilize_output:
        cmd.append('--no-stabilize-output')
    if stability_apply_to_context:
        cmd.append('--stability-apply-to-context')
    logger.info(
        "Running rollout checkpoint eval: "
        f"epoch={epoch_idx + 1}, metric_key={metric_key}, lookback={lookback}, block_len={block_len}, "
        f"feedback={feedback_source}, device={rollout_device}, "
        f"trade_fee_bps={trade_fee_bps:.4f}, "
        f"stabilize_output={stabilize_output}, "
        f"stability_apply_to_context={stability_apply_to_context}"
    )
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)

    with open(out_json, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    if metric_key not in metrics:
        raise KeyError(f"Rollout metrics missing key {metric_key!r}: {out_json}")
    return {
        'metric_key': metric_key,
        'metric_value': float(metrics[metric_key]),
        'metrics_json': out_json,
        'prediction_csv': out_csv,
    }


class CustomKlineDataset(Dataset):

    def __init__(
        self,
        data_path,
        data_type='train',
        lookback_window=90,
        predict_window=10,
        clip=5.0,
        seed=100,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sampling_strategy=None,
    ):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window + 1
        self.clip = clip
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        if sampling_strategy is None:
            self.sampling_strategy = 'independent' if data_type == 'train' else 'consecutive'
        else:
            self.sampling_strategy = str(sampling_strategy).strip().lower()
        if self.sampling_strategy not in {'independent', 'consecutive'}:
            raise ValueError(
                f"Unsupported sampling_strategy={self.sampling_strategy}. "
                "Use 'independent' or 'consecutive'."
            )
        
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        
        self.py_rng = random.Random(seed)
        
        self._load_and_preprocess_data()
        self._split_data_by_time()
        
        self.n_samples = len(self.data) - self.window + 1
            
        print(
            f"[{data_type.upper()}] Data length: {len(self.data)}, "
            f"Available samples: {self.n_samples}, Sampling: {self.sampling_strategy}"
        )
    
    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        self.timestamps = df['timestamps'].copy()
        
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
        
        self.data = df[self.feature_list + self.time_feature_list].copy()
        
        if self.data.isnull().any().any():
            print("Warning: Missing values found in data, performing forward fill")
            self.data = self.data.fillna(method='ffill')
        
        print(f"Original data time range: {self.timestamps.min()} to {self.timestamps.max()}")
        print(f"Original data total length: {len(df)} records")
    
    def _split_data_by_time(self):
        total_length = len(self.data)
        
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))
        
        if self.data_type in {'full', 'all'}:
            print(f"[{self.data_type.upper()}] Using full dataset without time split")
            print(f"[{self.data_type.upper()}] Full set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        elif self.data_type == 'train':
            self.data = self.data.iloc[:train_end].copy()
            self.timestamps = self.timestamps.iloc[:train_end].copy()
            print(f"[{self.data_type.upper()}] Training set: first {train_end} time points ({self.train_ratio})")
            print(f"[{self.data_type.upper()}] Training set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        elif self.data_type == 'val':
            self.data = self.data.iloc[train_end:val_end].copy()
            self.timestamps = self.timestamps.iloc[train_end:val_end].copy()
            print(f"[{self.data_type.upper()}] Validation set: time points {train_end+1} to {val_end} ({self.val_ratio})")
            print(f"[{self.data_type.upper()}] Validation set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        elif self.data_type == 'test':
            self.data = self.data.iloc[val_end:].copy()
            self.timestamps = self.timestamps.iloc[val_end:].copy()
            print(f"[{self.data_type.upper()}] Test set: after time point {val_end+1}")
            print(f"[{self.data_type.upper()}] Test set time range: {self.timestamps.min()} to {self.timestamps.max()}")
        
        print(f"[{self.data_type.upper()}] Data length after split: {len(self.data)} records")
    
    def set_epoch_seed(self, epoch):
        epoch_seed = self.seed + epoch
        self.py_rng.seed(epoch_seed)
        self.current_epoch = epoch
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        max_start = len(self.data) - self.window
        if max_start <= 0:
            raise ValueError("Data length insufficient to create samples")

        if self.sampling_strategy == 'independent':
            epoch = getattr(self, 'current_epoch', 0)
            # Deterministic hash-based sampling keeps windows independent of local index order
            # and stays stable under dataloader workers/DDP partitioning.
            hashed = (
                idx * 1103515245
                + (epoch + 1) * 12345
                + self.seed * 2654435761
            ) & 0x7FFFFFFF
            start_idx = hashed % (max_start + 1)
        else:
            start_idx = idx % (max_start + 1)
        
        end_idx = start_idx + self.window
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        x = window_data[self.feature_list].values.astype(np.float32)
        x_stamp = window_data[self.time_feature_list].values.astype(np.float32)

        # Normalize the full training window using statistics from the
        # historical context only, so future prediction rows cannot leak into
        # the scale seen by the model.
        context_len = min(max(int(self.lookback_window), 1), x.shape[0])
        context_x = x[:context_len]
        x_mean, x_std = np.mean(context_x, axis=0), np.std(context_x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)
        
        x_tensor = torch.from_numpy(x)
        x_stamp_tensor = torch.from_numpy(x_stamp)
        
        return x_tensor, x_stamp_tensor




def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Basemodel Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")

    train_data_path = getattr(config, 'train_data_path', None) or config.data_path
    val_data_path = getattr(config, 'val_data_path', None) or config.data_path
    train_data_mode = str(getattr(config, 'train_data_mode', 'train')).strip().lower()
    val_data_mode = str(getattr(config, 'val_data_mode', 'val')).strip().lower()
    
    train_dataset = CustomKlineDataset(
        data_path=train_data_path,
        data_type=train_data_mode,
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        sampling_strategy=getattr(config, 'train_sampling_strategy', 'independent'),
    )
    
    val_dataset = CustomKlineDataset(
        data_path=val_data_path,
        data_type=val_data_mode,
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        sampling_strategy=getattr(config, 'val_sampling_strategy', 'consecutive'),
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None
    train_shuffle = (train_sampler is None) and (train_dataset.sampling_strategy == 'independent')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
        print(f"Training data source: {train_data_path} ({train_data_mode})")
        print(f"Validation data source: {val_data_path} ({val_data_mode})")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def _configure_output_adapter(model, config, logger):
    use_adapter = bool(getattr(config, 'predictor_use_output_adapter', False))
    adapter_bias = bool(getattr(config, 'predictor_output_adapter_bias', True))
    adapter_residual = bool(getattr(config, 'predictor_output_adapter_residual', True))

    has_adapter = hasattr(model, 'output_adapter') and model.output_adapter is not None
    if use_adapter and not has_adapter:
        model.use_output_adapter = True
        model.output_adapter_bias = adapter_bias
        model.output_adapter_residual = adapter_residual
        if hasattr(model, "_hub_mixin_config") and isinstance(model._hub_mixin_config, dict):
            model._hub_mixin_config.update(
                {
                    "use_output_adapter": True,
                    "output_adapter_bias": adapter_bias,
                    "output_adapter_residual": adapter_residual,
                }
            )
        model.output_adapter = nn.Linear(model.d_model, model.d_model, bias=adapter_bias)
        model.output_adapter.to(next(model.parameters()).device)
        with torch.no_grad():
            nn.init.eye_(model.output_adapter.weight)
            if model.output_adapter.bias is not None:
                nn.init.zeros_(model.output_adapter.bias)
        logger.info(
            "Enabled predictor output adapter (identity init): "
            f"bias={adapter_bias}, residual={adapter_residual}"
        )
        return

    if has_adapter:
        if not getattr(model, 'use_output_adapter', True):
            model.use_output_adapter = True
        model.output_adapter_residual = adapter_residual
        if hasattr(model, "_hub_mixin_config") and isinstance(model._hub_mixin_config, dict):
            model._hub_mixin_config.update(
                {
                    "use_output_adapter": True,
                    "output_adapter_bias": (model.output_adapter.bias is not None),
                    "output_adapter_residual": model.output_adapter_residual,
                }
            )
        logger.info(
            "Using existing predictor output adapter: "
            f"bias={model.output_adapter.bias is not None}, residual={model.output_adapter_residual}"
        )
    else:
        logger.info("Predictor output adapter disabled")


def _resolve_optional_probability(raw_value, field_name: str):
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if not raw_value:
            return None

    value = float(raw_value)
    if not (0.0 <= value < 1.0):
        raise ValueError(f"{field_name} must be in [0, 1), got {value}")
    return value


def _apply_predictor_dropout_overrides(model, config, logger):
    ffn_dropout = _resolve_optional_probability(
        getattr(config, 'predictor_override_ffn_dropout_p', None),
        'predictor_override_ffn_dropout_p',
    )
    attn_dropout = _resolve_optional_probability(
        getattr(config, 'predictor_override_attn_dropout_p', None),
        'predictor_override_attn_dropout_p',
    )
    resid_dropout = _resolve_optional_probability(
        getattr(config, 'predictor_override_resid_dropout_p', None),
        'predictor_override_resid_dropout_p',
    )
    token_dropout = _resolve_optional_probability(
        getattr(config, 'predictor_override_token_dropout_p', None),
        'predictor_override_token_dropout_p',
    )

    applied = {}

    if ffn_dropout is not None:
        model.ffn_dropout_p = ffn_dropout
        for block in getattr(model, 'transformer', []):
            if hasattr(block, 'ffn') and hasattr(block.ffn, 'ffn_dropout'):
                block.ffn.ffn_dropout.p = ffn_dropout
        applied['ffn_dropout_p'] = ffn_dropout

    if attn_dropout is not None:
        model.attn_dropout_p = attn_dropout
        for block in getattr(model, 'transformer', []):
            if hasattr(block, 'self_attn'):
                block.self_attn.attn_dropout_p = attn_dropout
        dep_layer = getattr(model, 'dep_layer', None)
        cross_attn = getattr(dep_layer, 'cross_attn', None)
        if cross_attn is not None:
            cross_attn.attn_dropout_p = attn_dropout
        applied['attn_dropout_p'] = attn_dropout

    if resid_dropout is not None:
        model.resid_dropout_p = resid_dropout
        for block in getattr(model, 'transformer', []):
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'resid_dropout'):
                block.self_attn.resid_dropout.p = resid_dropout
        dep_layer = getattr(model, 'dep_layer', None)
        cross_attn = getattr(dep_layer, 'cross_attn', None)
        if cross_attn is not None and hasattr(cross_attn, 'resid_dropout'):
            cross_attn.resid_dropout.p = resid_dropout
        applied['resid_dropout_p'] = resid_dropout

    if token_dropout is not None:
        model.token_dropout_p = token_dropout
        if hasattr(model, 'token_drop'):
            model.token_drop.p = token_dropout
        applied['token_dropout_p'] = token_dropout

    if hasattr(model, '_hub_mixin_config') and isinstance(model._hub_mixin_config, dict):
        model._hub_mixin_config.update(applied)

    if applied:
        logger.info(
            "Applied predictor dropout overrides: "
            + ", ".join(f"{key}={value:.4f}" for key, value in applied.items())
        )
    else:
        logger.info("Predictor dropout overrides disabled; using checkpoint dropout values")


def _configure_trainable_parameters(model, config, logger):
    last_n = int(getattr(config, 'predictor_train_last_n_transformer_layers', -1))

    for p in model.parameters():
        p.requires_grad_(True)

    unfreezed_transformer = "all"
    if last_n >= 0:
        for p in model.parameters():
            p.requires_grad_(False)

        n_total = len(getattr(model, 'transformer', []))
        n_keep = min(max(last_n, 0), n_total)
        start_idx = n_total - n_keep

        if n_keep > 0:
            for block in model.transformer[start_idx:]:
                for p in block.parameters():
                    p.requires_grad_(True)
            unfreezed_transformer = f"{start_idx}-{n_total - 1}"
        else:
            unfreezed_transformer = "none"

        for module_name in ("norm", "dep_layer", "head", "output_adapter"):
            module = getattr(model, module_name, None)
            if module is None:
                continue
            for p in module.parameters():
                p.requires_grad_(True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    ratio = (100.0 * trainable_param_count / total_params) if total_params > 0 else 0.0
    logger.info(
        "Trainable predictor params: "
        f"{trainable_param_count:,}/{total_params:,} ({ratio:.2f}%). "
        f"predictor_train_last_n_transformer_layers={last_n}, "
        f"trainable_transformer_blocks={unfreezed_transformer}, "
        "always_trainable_modules=norm,dep_layer,head,output_adapter(if enabled)"
    )
    if not trainable_params:
        raise ValueError("No trainable predictor parameters after applying freeze policy")
    return trainable_params


def _parameter_lr_scale(name: str, model, layerwise_decay: float) -> float:
    layerwise_decay = float(layerwise_decay)
    if abs(layerwise_decay - 1.0) < 1e-12:
        return 1.0

    n_total = len(getattr(model, 'transformer', []))
    if name.startswith('transformer.'):
        parts = name.split('.')
        if len(parts) >= 2 and parts[1].isdigit():
            layer_idx = int(parts[1])
            exponent = max(n_total - 1 - layer_idx, 0)
            return float(layerwise_decay ** exponent)

    if name.startswith('embedding.') or name.startswith('time_emb.'):
        return float(layerwise_decay ** max(n_total, 0))

    return 1.0


def _should_exclude_from_weight_decay(name: str, param: torch.Tensor, config) -> bool:
    if not bool(getattr(config, 'predictor_weight_decay_exclude_bias_norm', False)):
        return False
    return param.ndim <= 1 or name.endswith('.bias')


def _build_predictor_optimizer(model, config, logger):
    base_lr = float(getattr(config, 'predictor_learning_rate', 0.0))
    weight_decay = float(getattr(config, 'adam_weight_decay', 0.0))
    layerwise_decay = float(getattr(config, 'predictor_layerwise_lr_decay', 1.0))
    if layerwise_decay <= 0.0:
        raise ValueError("predictor_layerwise_lr_decay must be > 0")

    grouped_params: dict[tuple[float, float], dict[str, object]] = defaultdict(
        lambda: {"params": [], "names": []}
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_scale = _parameter_lr_scale(name, model, layerwise_decay)
        lr = base_lr * lr_scale
        group_weight_decay = 0.0 if _should_exclude_from_weight_decay(name, param, config) else weight_decay
        group_key = (float(lr), float(group_weight_decay))
        grouped_params[group_key]["params"].append(param)
        grouped_params[group_key]["names"].append(name)

    if not grouped_params:
        raise ValueError("No optimizer parameter groups created for predictor")

    param_groups: list[dict[str, object]] = []
    summary_rows: list[str] = []
    for group_idx, ((lr, group_weight_decay), payload) in enumerate(
        sorted(grouped_params.items(), key=lambda item: (-item[0][0], item[0][1]))
    ):
        params = payload["params"]
        names = payload["names"]
        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": group_weight_decay,
            }
        )
        param_count = sum(param.numel() for param in params)
        preview_names = ", ".join(str(name) for name in names[:3])
        if len(names) > 3:
            preview_names += ", ..."
        summary_rows.append(
            f"group={group_idx}, lr={lr:.8g}, wd={group_weight_decay:.6g}, "
            f"tensors={len(params)}, params={param_count:,}, names={preview_names}"
        )

    logger.info(
        "Predictor optimizer groups:\n" + "\n".join(summary_rows)
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(config.adam_beta1, config.adam_beta2),
    )
    return optimizer


def _compute_pretrained_anchor_loss(model, reference_state: dict[str, torch.Tensor] | None):
    if not reference_state:
        return None, 0, 0

    total_sq = None
    total_elems = 0
    anchored_params = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith('output_adapter.'):
            continue
        ref_tensor = reference_state.get(name)
        if ref_tensor is None or tuple(ref_tensor.shape) != tuple(param.shape):
            continue
        diff = param - ref_tensor.to(device=param.device, dtype=param.dtype)
        sq = diff.pow(2).sum()
        total_sq = sq if total_sq is None else total_sq + sq
        total_elems += int(param.numel())
        anchored_params += 1

    if total_sq is None or total_elems <= 0:
        return None, 0, 0
    return total_sq / float(total_elems), anchored_params, total_elems


def _resolve_scheduler_warmup_steps(config, total_optimizer_steps: int) -> int:
    total_optimizer_steps = max(int(total_optimizer_steps), 1)
    warmup_steps = int(getattr(config, 'predictor_scheduler_warmup_steps', 0))
    if warmup_steps <= 0:
        warmup_ratio = float(getattr(config, 'predictor_scheduler_warmup_ratio', 0.0))
        if warmup_ratio > 0.0:
            warmup_steps = int(round(total_optimizer_steps * warmup_ratio))
    return max(0, min(int(warmup_steps), max(total_optimizer_steps - 1, 0)))


def _build_predictor_scheduler(optimizer, config, steps_per_epoch: int, total_epochs: int, higher_is_better: bool):
    scheduler_name = str(getattr(config, 'predictor_scheduler_name', 'onecycle')).strip().lower()
    scheduler_name = scheduler_name.replace('-', '_')
    if scheduler_name in {'', 'default'}:
        scheduler_name = 'onecycle'

    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_epochs = max(int(total_epochs), 1)
    total_optimizer_steps = max(steps_per_epoch * total_epochs, 1)
    base_lr = float(optimizer.param_groups[0]['lr'])
    warmup_steps = _resolve_scheduler_warmup_steps(config, total_optimizer_steps)
    warmup_start_factor = float(getattr(config, 'predictor_scheduler_warmup_start_factor', 0.1))
    warmup_start_factor = float(np.clip(warmup_start_factor, 1e-4, 1.0))
    min_lr_ratio = float(getattr(config, 'predictor_scheduler_min_lr_ratio', 0.0))
    min_lr_ratio = float(np.clip(min_lr_ratio, 0.0, 1.0))
    plateau_metric_name = str(
        getattr(config, 'predictor_scheduler_plateau_metric', 'checkpoint_metric')
    ).strip().lower()

    if scheduler_name == 'none':
        return None, 'none', scheduler_name, plateau_metric_name

    if scheduler_name == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            pct_start=float(getattr(config, 'predictor_scheduler_onecycle_pct_start', 0.03)),
            div_factor=float(getattr(config, 'predictor_scheduler_onecycle_div_factor', 10.0)),
            final_div_factor=float(getattr(config, 'predictor_scheduler_onecycle_final_div_factor', 100.0)),
        )
        return scheduler, 'batch', scheduler_name, plateau_metric_name

    if scheduler_name in {'constant', 'constant_warmup'}:
        if warmup_steps <= 0:
            return None, 'none', scheduler_name, plateau_metric_name

        def _constant_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                progress = float(step_idx + 1) / float(max(warmup_steps, 1))
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_constant_lambda)
        return scheduler, 'batch', scheduler_name, plateau_metric_name

    if scheduler_name in {'linear', 'linear_warmup', 'linear_decay'}:
        def _linear_lambda(step_idx: int) -> float:
            current_step = min(int(step_idx + 1), total_optimizer_steps)
            if warmup_steps > 0 and current_step <= warmup_steps:
                progress = float(current_step) / float(max(warmup_steps, 1))
                return warmup_start_factor + (1.0 - warmup_start_factor) * progress

            decay_steps = max(total_optimizer_steps - warmup_steps, 1)
            decay_progress = float(current_step - warmup_steps) / float(decay_steps)
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * decay_progress)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_linear_lambda)
        return scheduler, 'batch', scheduler_name, plateau_metric_name

    if scheduler_name in {'cosine', 'cosine_annealing'}:
        cosine_steps = max(total_optimizer_steps - warmup_steps, 1)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=base_lr * min_lr_ratio,
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler
        return scheduler, 'batch', scheduler_name, plateau_metric_name

    if scheduler_name in {'cosine_restarts', 'cosine_warm_restarts'}:
        restart_t0 = int(getattr(config, 'predictor_scheduler_restart_t0', 0))
        if restart_t0 <= 0:
            restart_t0 = max(steps_per_epoch, 1)
        restart_t_mult = max(int(getattr(config, 'predictor_scheduler_restart_t_mult', 1)), 1)
        restart_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=restart_t0,
            T_mult=restart_t_mult,
            eta_min=base_lr * min_lr_ratio,
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, restart_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = restart_scheduler
        return scheduler, 'batch', scheduler_name, plateau_metric_name

    if scheduler_name == 'plateau':
        plateau_factor = float(getattr(config, 'predictor_scheduler_plateau_factor', 0.5))
        plateau_factor = float(np.clip(plateau_factor, 1e-4, 0.9999))
        plateau_patience = max(int(getattr(config, 'predictor_scheduler_plateau_patience', 2)), 0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max' if higher_is_better else 'min',
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=base_lr * min_lr_ratio,
        )
        return scheduler, 'epoch', scheduler_name, plateau_metric_name

    raise ValueError(
        f"Unsupported predictor_scheduler_name={scheduler_name}. "
        "Use one of: none, onecycle, constant, linear, cosine, cosine_restarts, plateau."
    )


def _resolve_epoch_scheduler_metric(
    metric_name: str,
    *,
    avg_val_loss: float,
    avg_val_cons_dir_acc: float,
    rollout_metric_value: float | None,
    current_checkpoint_value: float | None,
) -> float | None:
    metric_name = str(metric_name).strip().lower()
    if metric_name in {'checkpoint', 'checkpoint_metric', 'auto'}:
        return current_checkpoint_value
    if metric_name in {'val_loss', 'loss'}:
        return avg_val_loss
    if metric_name in {'val_cons_dir_acc', 'cons_dir_acc'}:
        return avg_val_cons_dir_acc
    if metric_name in {'rollout', 'rollout_metric'}:
        return rollout_metric_value
    raise ValueError(
        f"Unsupported predictor_scheduler_plateau_metric={metric_name}. "
        "Use checkpoint_metric, val_loss, val_cons_dir_acc, or rollout_metric."
    )


def train_model(model, tokenizer, device, config, save_dir, logger, pretrained_reference_state=None):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0

    objective_mode = str(getattr(config, 'predictor_objective_mode', 'hybrid')).strip().lower()
    checkpoint_metric = _normalize_checkpoint_metric(getattr(config, 'predictor_checkpoint_metric', 'val_loss'))
    if checkpoint_metric not in {'val_loss', 'val_cons_dir_acc', 'rollout_metric'}:
        raise ValueError(
            f"Unsupported predictor_checkpoint_metric={checkpoint_metric}. "
            "Use 'val_loss', 'val_cons_dir_acc', or 'rollout_metric'."
        )

    ce_weight = float(getattr(config, 'predictor_ce_loss_weight', 1.0))
    dir_weight = float(getattr(config, 'predictor_directional_loss_weight', 0.0))
    cons_dir_weight = float(getattr(config, 'predictor_consecutive_directional_loss_weight', 0.0))
    cons_dir_pos_weight = float(getattr(config, 'predictor_consecutive_pos_weight', 1.0))
    cons_dir_focal_gamma = float(getattr(config, 'predictor_consecutive_focal_gamma', 0.0))
    cons_dir_label_smoothing = float(getattr(config, 'predictor_consecutive_label_smoothing', 0.0))
    cons_dir_negative_weight = float(getattr(config, 'predictor_consecutive_negative_weight', 1.0))
    cons_dir_false_positive_weight = float(
        getattr(config, 'predictor_consecutive_false_positive_weight', 0.0)
    )
    cons_dir_bias_penalty_weight = float(
        getattr(config, 'predictor_consecutive_bias_penalty_weight', 0.0)
    )
    cons_dir_magnitude_weight = float(getattr(config, 'predictor_consecutive_magnitude_weight', 0.0))
    cons_dir_magnitude_mode = str(
        getattr(config, 'predictor_consecutive_magnitude_mode', 'relative_return')
    ).strip().lower()
    cons_dir_magnitude_scale = float(getattr(config, 'predictor_consecutive_magnitude_scale', 0.01))
    cons_dir_magnitude_power = float(getattr(config, 'predictor_consecutive_magnitude_power', 1.0))
    cons_dir_magnitude_cap = float(getattr(config, 'predictor_consecutive_magnitude_cap', 5.0))
    mse_weight = float(getattr(config, 'predictor_mse_loss_weight', 0.0))
    accumulation_steps = max(int(getattr(config, 'accumulation_steps', 1)), 1)
    rollout_eval_interval = max(int(getattr(config, 'predictor_rollout_eval_interval_epochs', 1)), 1)
    layerwise_lr_decay = float(getattr(config, 'predictor_layerwise_lr_decay', 1.0))
    pretrained_anchor_weight = float(getattr(config, 'predictor_pretrained_anchor_weight', 0.0))
    ema_decay = float(getattr(config, 'predictor_ema_decay', 0.0))
    ema_start_step = max(int(getattr(config, 'predictor_ema_start_step', 0)), 0)
    ema_eval = bool(getattr(config, 'predictor_ema_eval', False))
    early_stopping_patience = max(int(getattr(config, 'predictor_early_stopping_patience', 0)), 0)
    early_stopping_min_delta = float(getattr(config, 'predictor_early_stopping_min_delta', 0.0))

    if objective_mode in {"consecutive_path", "consecutive_path_only"}:
        # Align optimization to directional_accuracy_close_vs_prev_close_consecutive_path.
        if cons_dir_weight <= 0.0:
            cons_dir_weight = 1.0
        if objective_mode == "consecutive_path_only":
            ce_weight = 0.0
            dir_weight = 0.0
            mse_weight = 0.0

    use_aux_losses = (dir_weight > 0.0) or (cons_dir_weight > 0.0) or (mse_weight > 0.0)
    logger.info(
        "Predictor objective weights: "
        f"CE={ce_weight:.4f}, DIR={dir_weight:.4f}, CONS_DIR={cons_dir_weight:.4f}, MSE={mse_weight:.4f}; "
        f"CONS_DIR_POS_W={cons_dir_pos_weight:.4f}, CONS_DIR_FOCAL_GAMMA={cons_dir_focal_gamma:.4f}, "
        f"CONS_DIR_LABEL_SMOOTH={cons_dir_label_smoothing:.4f}, "
        f"CONS_DIR_NEG_W={cons_dir_negative_weight:.4f}, "
        f"CONS_DIR_FALSE_POS_W={cons_dir_false_positive_weight:.4f}, "
        f"CONS_DIR_BIAS_PENALTY_W={cons_dir_bias_penalty_weight:.4f}, "
        f"CONS_DIR_MAG_W={cons_dir_magnitude_weight:.4f}, CONS_DIR_MAG_MODE={cons_dir_magnitude_mode}, "
        f"CONS_DIR_MAG_SCALE={cons_dir_magnitude_scale:.6f}, "
        f"CONS_DIR_MAG_POWER={cons_dir_magnitude_power:.4f}, CONS_DIR_MAG_CAP={cons_dir_magnitude_cap:.4f}; "
        f"MODE={objective_mode}"
    )
    higher_is_better = checkpoint_metric in {'val_cons_dir_acc', 'rollout_metric'}
    logger.info(
        "Checkpoint + optimizer settings: "
        f"checkpoint_metric={checkpoint_metric}, accumulation_steps={accumulation_steps}, "
        f"scheduler={getattr(config, 'predictor_scheduler_name', 'onecycle')}, "
        f"layerwise_lr_decay={layerwise_lr_decay:.4f}, "
        f"pretrained_anchor_weight={pretrained_anchor_weight:.6f}, "
        f"ema_decay={ema_decay:.6f}, ema_start_step={ema_start_step}, ema_eval={ema_eval}, "
        f"early_stopping_patience={early_stopping_patience}, early_stopping_min_delta={early_stopping_min_delta:.6f}, "
        f"scheduled_sampling_max_rate={float(getattr(config, 'predictor_scheduled_sampling_max_rate', 0.0)):.4f}, "
        f"scheduled_sampling_strategy={getattr(config, 'predictor_scheduled_sampling_strategy', 'tail')}, "
        f"rollout_eval_interval_epochs={rollout_eval_interval}"
    )
    if checkpoint_metric == 'rollout_metric':
        logger.info(
            "Rollout checkpoint eval config: "
            f"metric_key={getattr(config, 'predictor_rollout_metric_key', 'directional_accuracy_close_vs_prev_close_consecutive_path')}, "
            f"train_context={getattr(config, 'predictor_rollout_train_context_path', '')}, "
            f"eval_path={getattr(config, 'predictor_rollout_eval_path', '')}"
        )

    _configure_output_adapter(model, config, logger)
    trainable_params = _configure_trainable_parameters(model, config, logger)
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    del trainable_params
    optimizer = _build_predictor_optimizer(model, config, logger)
    optimizer_steps_per_epoch = max(1, int(np.ceil(len(train_loader) / accumulation_steps)))
    scheduler, scheduler_step_mode, scheduler_name, plateau_metric_name = _build_predictor_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=optimizer_steps_per_epoch,
        total_epochs=config.basemodel_epochs,
        higher_is_better=higher_is_better,
    )
    logger.info(
        "Resolved predictor LR schedule: "
        f"name={scheduler_name}, step_mode={scheduler_step_mode}, "
        f"warmup_steps={_resolve_scheduler_warmup_steps(config, optimizer_steps_per_epoch * config.basemodel_epochs)}, "
        f"min_lr_ratio={float(getattr(config, 'predictor_scheduler_min_lr_ratio', 0.0)):.6f}"
    )
    if pretrained_anchor_weight > 0.0 and pretrained_reference_state:
        ref_match_count = sum(
            1
            for name, param in model.named_parameters()
            if param.requires_grad and name in pretrained_reference_state and not name.startswith('output_adapter.')
        )
        logger.info(
            "Pretrained anchor regularization enabled: "
            f"weight={pretrained_anchor_weight:.6f}, matched_trainable_params={ref_match_count}"
        )
    elif pretrained_anchor_weight > 0.0:
        logger.warning(
            "predictor_pretrained_anchor_weight > 0 but no pretrained reference state was provided; "
            "anchor regularization will be skipped"
        )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    ema_helper = None
    if ema_decay > 0.0:
        ema_helper = ModelEma(model.module if use_ddp else model, ema_decay)
        logger.info(
            "EMA enabled: "
            f"decay={ema_decay:.6f}, start_step={ema_start_step}, eval_with_ema={ema_eval}"
        )

    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    best_val_loss = float('inf')
    best_checkpoint_value = -float('inf') if higher_is_better else float('inf')
    best_checkpoint_epoch = -1
    batch_idx_global = 0
    optimizer_step_global = 0
    epochs_without_improvement = 0
    train_start_time = time.time()
    step_rows: list[dict[str, float]] = []
    epoch_rows: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    
    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()
        scheduled_sampling_rate = _scheduled_sampling_rate(config, epoch)
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        epoch_train_ce = 0.0
        epoch_train_dir = 0.0
        epoch_train_cons_dir = 0.0
        epoch_train_mse = 0.0
        epoch_train_anchor = 0.0
        epoch_train_dir_acc = 0.0
        epoch_train_cons_dir_acc = 0.0
        epoch_train_sched_replace = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            token_in_model, _, batch_sched_replace = _apply_scheduled_sampling(
                model=model,
                use_ddp=use_ddp,
                token_in=token_in,
                stamp=batch_x_stamp[:, :-1, :],
                config=config,
                epoch_idx=epoch,
            )
            
            logits = (model.module if use_ddp else model)(
                token_in_model[0],
                token_in_model[1],
                batch_x_stamp[:, :-1, :],
            )
            ce_loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])

            if use_aux_losses:
                pred_seq = _decode_soft_predictions(tokenizer, logits[0], logits[1])
                target_seq = batch_x[:, 1:, :]
                context_close = batch_x[:, 0, PRICE_CLOSE_COL]
                context_seq_len = min(batch_x.size(1), max(int(getattr(config, 'lookback_window', 0)), 1) + 1)
                context_close_seq = batch_x[:, :context_seq_len, PRICE_CLOSE_COL]

                if dir_weight > 0.0:
                    dir_loss, dir_acc = _directional_loss_and_accuracy(pred_seq, target_seq)
                else:
                    dir_loss = ce_loss.new_zeros(())
                    dir_acc = ce_loss.new_zeros(())

                if cons_dir_weight > 0.0:
                    cons_dir_loss, cons_dir_acc = _consecutive_directional_loss_and_accuracy(
                        pred_seq=pred_seq,
                        target_seq=target_seq,
                        context_close=context_close,
                        context_close_seq=context_close_seq,
                        pos_weight=cons_dir_pos_weight,
                        focal_gamma=cons_dir_focal_gamma,
                        label_smoothing=cons_dir_label_smoothing,
                        negative_weight=cons_dir_negative_weight,
                        false_positive_weight=cons_dir_false_positive_weight,
                        bias_penalty_weight=cons_dir_bias_penalty_weight,
                        magnitude_weight=cons_dir_magnitude_weight,
                        magnitude_mode=cons_dir_magnitude_mode,
                        magnitude_scale=cons_dir_magnitude_scale,
                        magnitude_power=cons_dir_magnitude_power,
                        magnitude_cap=cons_dir_magnitude_cap,
                    )
                else:
                    cons_dir_loss = ce_loss.new_zeros(())
                    cons_dir_acc = ce_loss.new_zeros(())

                if mse_weight > 0.0:
                    mse_loss = F.mse_loss(pred_seq, target_seq)
                else:
                    mse_loss = ce_loss.new_zeros(())
            else:
                dir_loss = ce_loss.new_zeros(())
                dir_acc = ce_loss.new_zeros(())
                cons_dir_loss = ce_loss.new_zeros(())
                cons_dir_acc = ce_loss.new_zeros(())
                mse_loss = ce_loss.new_zeros(())

            loss = (
                ce_weight * ce_loss
                + dir_weight * dir_loss
                + cons_dir_weight * cons_dir_loss
                + mse_weight * mse_loss
            )
            if pretrained_anchor_weight > 0.0:
                anchor_loss, _, _ = _compute_pretrained_anchor_loss(
                    model.module if use_ddp else model,
                    pretrained_reference_state,
                )
                if anchor_loss is None:
                    anchor_loss = ce_loss.new_zeros(())
                loss = loss + pretrained_anchor_weight * anchor_loss
            else:
                anchor_loss = ce_loss.new_zeros(())
            
            (loss / accumulation_steps).backward()
            should_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if should_step:
                torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
                optimizer.step()
                optimizer_step_global += 1
                if ema_helper is not None and optimizer_step_global >= ema_start_step:
                    ema_helper.update(model.module if use_ddp else model)
                if scheduler is not None and scheduler_step_mode == 'batch':
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            epoch_train_loss += loss.item()
            epoch_train_ce += ce_loss.item()
            epoch_train_dir += dir_loss.item()
            epoch_train_cons_dir += cons_dir_loss.item()
            epoch_train_mse += mse_loss.item()
            epoch_train_anchor += anchor_loss.item()
            epoch_train_dir_acc += dir_acc.item()
            epoch_train_cons_dir_acc += cons_dir_acc.item()
            epoch_train_sched_replace += batch_sched_replace
            train_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
                if use_aux_losses:
                    aux_msg = (
                        f"  - CE: {ce_loss.item():.4f}\n"
                        f"  - Dir Loss: {dir_loss.item():.4f}\n"
                        f"  - Cons Dir Loss: {cons_dir_loss.item():.4f}\n"
                        f"  - MSE: {mse_loss.item():.4f}\n"
                        f"  - Anchor Loss: {anchor_loss.item():.4f}\n"
                        f"  - Dir Acc: {dir_acc.item()*100.0:.2f}%\n"
                        f"  - Cons Dir Acc: {cons_dir_acc.item()*100.0:.2f}%\n"
                        f"  - Scheduled Sampling Rate: {scheduled_sampling_rate*100.0:.2f}%\n"
                        f"  - Scheduled Sampling Replace: {batch_sched_replace*100.0:.2f}%"
                    )
                    logger.info(aux_msg)
                    if rank == 0:
                        print(aux_msg)
                if rank == 0:
                    step_rows.append(
                        {
                            "epoch": float(epoch + 1),
                            "step_in_epoch": float(batch_idx + 1),
                            "global_step": float(batch_idx_global + 1),
                            "lr": float(lr),
                            "loss_total": float(loss.item()),
                            "ce_loss": float(ce_loss.item()),
                            "dir_loss": float(dir_loss.item()),
                            "cons_dir_loss": float(cons_dir_loss.item()),
                            "mse_loss": float(mse_loss.item()),
                            "anchor_loss": float(anchor_loss.item()),
                            "dir_acc_pct": float(dir_acc.item() * 100.0),
                            "cons_dir_acc_pct": float(cons_dir_acc.item() * 100.0),
                            "scheduled_sampling_rate_pct": float(scheduled_sampling_rate * 100.0),
                            "scheduled_sampling_replace_pct": float(batch_sched_replace * 100.0),
                        }
                    )
            
            batch_idx_global += 1
        
        ema_weights_applied = False
        eval_target_model = model.module if use_ddp else model
        if (
            ema_helper is not None
            and ema_eval
            and optimizer_step_global >= ema_start_step
            and ema_helper.has_state()
        ):
            ema_helper.store(eval_target_model)
            ema_helper.copy_to(eval_target_model)
            ema_weights_applied = True
            logger.info(f"Epoch {epoch+1}: evaluating with EMA weights")

        model.eval()
        val_loss = 0.0
        val_ce = 0.0
        val_dir = 0.0
        val_cons_dir = 0.0
        val_mse = 0.0
        val_dir_acc = 0.0
        val_cons_dir_acc = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                logits = (model.module if use_ddp else model)(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                ce_loss, _, _ = (model.module if use_ddp else model).head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                if use_aux_losses:
                    pred_seq = _decode_soft_predictions(tokenizer, logits[0], logits[1])
                    target_seq = batch_x[:, 1:, :]
                    context_close = batch_x[:, 0, PRICE_CLOSE_COL]
                    context_seq_len = min(batch_x.size(1), max(int(getattr(config, 'lookback_window', 0)), 1) + 1)
                    context_close_seq = batch_x[:, :context_seq_len, PRICE_CLOSE_COL]

                    if dir_weight > 0.0:
                        dir_loss, dir_acc = _directional_loss_and_accuracy(pred_seq, target_seq)
                    else:
                        dir_loss = ce_loss.new_zeros(())
                        dir_acc = ce_loss.new_zeros(())

                    if cons_dir_weight > 0.0:
                        cons_dir_loss, cons_dir_acc = _consecutive_directional_loss_and_accuracy(
                            pred_seq=pred_seq,
                            target_seq=target_seq,
                            context_close=context_close,
                            context_close_seq=context_close_seq,
                            pos_weight=cons_dir_pos_weight,
                            focal_gamma=cons_dir_focal_gamma,
                            label_smoothing=cons_dir_label_smoothing,
                            negative_weight=cons_dir_negative_weight,
                            false_positive_weight=cons_dir_false_positive_weight,
                            bias_penalty_weight=cons_dir_bias_penalty_weight,
                            magnitude_weight=cons_dir_magnitude_weight,
                            magnitude_mode=cons_dir_magnitude_mode,
                            magnitude_scale=cons_dir_magnitude_scale,
                            magnitude_power=cons_dir_magnitude_power,
                            magnitude_cap=cons_dir_magnitude_cap,
                        )
                    else:
                        cons_dir_loss = ce_loss.new_zeros(())
                        cons_dir_acc = ce_loss.new_zeros(())

                    if mse_weight > 0.0:
                        mse_loss = F.mse_loss(pred_seq, target_seq)
                    else:
                        mse_loss = ce_loss.new_zeros(())
                else:
                    dir_loss = ce_loss.new_zeros(())
                    dir_acc = ce_loss.new_zeros(())
                    cons_dir_loss = ce_loss.new_zeros(())
                    cons_dir_acc = ce_loss.new_zeros(())
                    mse_loss = ce_loss.new_zeros(())

                loss = (
                    ce_weight * ce_loss
                    + dir_weight * dir_loss
                    + cons_dir_weight * cons_dir_loss
                    + mse_weight * mse_loss
                )
                
                val_loss += loss.item()
                val_ce += ce_loss.item()
                val_dir += dir_loss.item()
                val_cons_dir += cons_dir_loss.item()
                val_mse += mse_loss.item()
                val_dir_acc += dir_acc.item()
                val_cons_dir_acc += cons_dir_acc.item()
                val_batches += 1
        
        if use_ddp:
            tensor_sum = torch.tensor(
                [
                    epoch_train_loss,
                    train_batches,
                    val_loss,
                    val_batches,
                    epoch_train_ce,
                    epoch_train_dir,
                    epoch_train_cons_dir,
                    epoch_train_mse,
                    epoch_train_anchor,
                    epoch_train_dir_acc,
                    epoch_train_cons_dir_acc,
                    val_ce,
                    val_dir,
                    val_cons_dir,
                    val_mse,
                    val_dir_acc,
                    val_cons_dir_acc,
                ],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            train_ce_all = tensor_sum[4].item()
            train_dir_all = tensor_sum[5].item()
            train_cons_dir_all = tensor_sum[6].item()
            train_mse_all = tensor_sum[7].item()
            train_anchor_all = tensor_sum[8].item()
            train_dir_acc_all = tensor_sum[9].item()
            train_cons_dir_acc_all = tensor_sum[10].item()
            val_ce_all = tensor_sum[11].item()
            val_dir_all = tensor_sum[12].item()
            val_cons_dir_all = tensor_sum[13].item()
            val_mse_all = tensor_sum[14].item()
            val_dir_acc_all = tensor_sum[15].item()
            val_cons_dir_acc_all = tensor_sum[16].item()
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_train_ce = (train_ce_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_dir = (train_dir_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_cons_dir = (train_cons_dir_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_mse = (train_mse_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_anchor = (train_anchor_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_dir_acc = (train_dir_acc_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_train_cons_dir_acc = (train_cons_dir_acc_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_ce = (val_ce_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_val_dir = (val_dir_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_val_cons_dir = (val_cons_dir_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_val_mse = (val_mse_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_val_dir_acc = (val_dir_acc_all / val_batches_all) if val_batches_all > 0 else 0.0
            avg_val_cons_dir_acc = (val_cons_dir_acc_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            avg_train_ce = epoch_train_ce / train_batches if train_batches > 0 else 0
            avg_train_dir = epoch_train_dir / train_batches if train_batches > 0 else 0
            avg_train_cons_dir = epoch_train_cons_dir / train_batches if train_batches > 0 else 0
            avg_train_mse = epoch_train_mse / train_batches if train_batches > 0 else 0
            avg_train_anchor = epoch_train_anchor / train_batches if train_batches > 0 else 0
            avg_train_dir_acc = epoch_train_dir_acc / train_batches if train_batches > 0 else 0
            avg_train_cons_dir_acc = epoch_train_cons_dir_acc / train_batches if train_batches > 0 else 0
            avg_val_ce = val_ce / val_batches if val_batches > 0 else 0
            avg_val_dir = val_dir / val_batches if val_batches > 0 else 0
            avg_val_cons_dir = val_cons_dir / val_batches if val_batches > 0 else 0
            avg_val_mse = val_mse / val_batches if val_batches > 0 else 0
            avg_val_dir_acc = val_dir_acc / val_batches if val_batches > 0 else 0
            avg_val_cons_dir_acc = val_cons_dir_acc / val_batches if val_batches > 0 else 0
       
        avg_sched_replace = epoch_train_sched_replace / train_batches if train_batches > 0 else 0.0
        rollout_metric_value = None
        rollout_metric_key = str(
            getattr(
                config,
                'predictor_rollout_metric_key',
                'directional_accuracy_close_vs_prev_close_consecutive_path',
            )
        )
        if checkpoint_metric == 'rollout_metric' and (
            ((epoch + 1) % rollout_eval_interval == 0) or ((epoch + 1) == config.basemodel_epochs)
        ):
            if use_ddp:
                dist.barrier()
            if rank == 0:
                rollout_eval = _run_rollout_checkpoint_eval(
                    model=(model.module if use_ddp else model),
                    tokenizer=tokenizer,
                    device=device,
                    config=config,
                    epoch_idx=epoch,
                    logger=logger,
                )
                rollout_metric_value = float(rollout_eval['metric_value'])
                logger.info(
                    f"Epoch {epoch+1} rollout metric {rollout_eval['metric_key']}: {rollout_metric_value:.6f}"
                )
            if use_ddp:
                shared_value = torch.tensor(
                    [rollout_metric_value if rollout_metric_value is not None else -1e38],
                    dtype=torch.float64,
                    device=device,
                )
                dist.broadcast(shared_value, src=0)
                rollout_metric_value = float(shared_value.item())
                if rollout_metric_value <= -1e37:
                    rollout_metric_value = None
                dist.barrier()

        if checkpoint_metric == 'val_loss':
            current_checkpoint_value = avg_val_loss
        elif checkpoint_metric == 'val_cons_dir_acc':
            current_checkpoint_value = avg_val_cons_dir_acc
        else:
            current_checkpoint_value = rollout_metric_value

        if scheduler is not None and scheduler_step_mode == 'epoch':
            scheduler_metric_value = _resolve_epoch_scheduler_metric(
                plateau_metric_name,
                avg_val_loss=avg_val_loss,
                avg_val_cons_dir_acc=avg_val_cons_dir_acc,
                rollout_metric_value=rollout_metric_value,
                current_checkpoint_value=current_checkpoint_value,
            )
            if scheduler_metric_value is not None:
                scheduler.step(scheduler_metric_value)

        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n")
        if use_aux_losses:
            epoch_summary += (
                       f"Train CE: {avg_train_ce:.4f}, Train Dir: {avg_train_dir:.4f}, Train ConsDir: {avg_train_cons_dir:.4f}, Train MSE: {avg_train_mse:.4f}, Train Anchor: {avg_train_anchor:.4f}, Train DirAcc: {avg_train_dir_acc*100.0:.2f}%, Train ConsDirAcc: {avg_train_cons_dir_acc*100.0:.2f}%\n"
                       f"Val CE: {avg_val_ce:.4f}, Val Dir: {avg_val_dir:.4f}, Val ConsDir: {avg_val_cons_dir:.4f}, Val MSE: {avg_val_mse:.4f}, Val DirAcc: {avg_val_dir_acc*100.0:.2f}%, Val ConsDirAcc: {avg_val_cons_dir_acc*100.0:.2f}%\n"
            )
        epoch_summary += (
                       f"Scheduled Sampling Rate: {scheduled_sampling_rate*100.0:.2f}%\n"
                       f"Scheduled Sampling Replace: {avg_sched_replace*100.0:.2f}%\n"
                       f"Current LR: {optimizer.param_groups[0]['lr']:.8f}\n")
        if ema_weights_applied:
            epoch_summary += "Eval Weights: EMA\n"
        if rollout_metric_value is not None:
            epoch_summary += f"Rollout {rollout_metric_key}: {rollout_metric_value:.6f}\n"
        if current_checkpoint_value is not None:
            epoch_summary += f"Checkpoint Metric ({checkpoint_metric}): {current_checkpoint_value:.6f}\n"
        epoch_summary += (
                       f"Epoch Time: {epoch_time:.2f} seconds\n"
                       f"Total Training Time: {time.time() - train_start_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
            epoch_rows.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(avg_train_loss),
                    "val_loss": float(avg_val_loss),
                    "train_ce_loss": float(avg_train_ce),
                    "train_dir_loss": float(avg_train_dir),
                    "train_cons_dir_loss": float(avg_train_cons_dir),
                    "train_mse_loss": float(avg_train_mse),
                    "train_anchor_loss": float(avg_train_anchor),
                    "train_dir_acc_pct": float(avg_train_dir_acc * 100.0),
                    "train_cons_dir_acc_pct": float(avg_train_cons_dir_acc * 100.0),
                    "val_ce_loss": float(avg_val_ce),
                    "val_dir_loss": float(avg_val_dir),
                    "val_cons_dir_loss": float(avg_val_cons_dir),
                    "val_mse_loss": float(avg_val_mse),
                    "val_dir_acc_pct": float(avg_val_dir_acc * 100.0),
                    "val_cons_dir_acc_pct": float(avg_val_cons_dir_acc * 100.0),
                    "scheduled_sampling_rate_pct": float(scheduled_sampling_rate * 100.0),
                    "scheduled_sampling_replace_pct": float(avg_sched_replace * 100.0),
                    "rollout_metric": float(rollout_metric_value) if rollout_metric_value is not None else float('nan'),
                    "checkpoint_metric_value": float(current_checkpoint_value) if current_checkpoint_value is not None else float('nan'),
                    "eval_used_ema": float(1.0 if ema_weights_applied else 0.0),
                    "epoch_time_sec": float(epoch_time),
                    "elapsed_total_sec": float(time.time() - train_start_time),
                    "best_val_loss_so_far": float(min(best_val_loss, avg_val_loss)),
                }
            )
            metrics_dir = getattr(config, "metrics_dir", os.path.join(config.base_save_path, "metrics"))
            save_training_dashboard(
                out_dir=metrics_dir,
                prefix="basemodel",
                step_rows=step_rows,
                epoch_rows=epoch_rows,
                logger=logger,
            )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        checkpoint_improved = _metric_improved(
            current_checkpoint_value,
            best_checkpoint_value,
            higher_is_better=higher_is_better,
            min_delta=early_stopping_min_delta,
        )

        if checkpoint_improved:
            best_checkpoint_value = current_checkpoint_value
            best_checkpoint_epoch = epoch + 1
            epochs_without_improvement = 0
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = (
                    f"Best model saved to: {model_save_path} "
                    f"({checkpoint_metric}={best_checkpoint_value:.6f}, epoch={best_checkpoint_epoch})"
                )
                logger.info(save_msg)
                print(save_msg)
        elif current_checkpoint_value is not None:
            epochs_without_improvement += 1

        if ema_weights_applied:
            ema_helper.restore(eval_target_model)

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "Early stopping triggered: "
                f"patience={early_stopping_patience}, "
                f"best_checkpoint_value={best_checkpoint_value:.6f}, "
                f"best_checkpoint_epoch={best_checkpoint_epoch}"
            )
            if rank == 0:
                print(
                    "Early stopping triggered: "
                    f"patience={early_stopping_patience}, best epoch={best_checkpoint_epoch}"
                )
            break
    if rank == 0:
        metrics_dir = getattr(config, "metrics_dir", os.path.join(config.base_save_path, "metrics"))
        dashboard = save_training_dashboard(
            out_dir=metrics_dir,
            prefix="basemodel",
            step_rows=step_rows,
            epoch_rows=epoch_rows,
            logger=logger,
        )
        logger.info(f"Basemodel charts and metrics saved: {json.dumps(dashboard, indent=2)}")
        logger.info(
            "Training complete: "
            f"best_val_loss={best_val_loss:.6f}, "
            f"best_checkpoint_metric={checkpoint_metric}, "
            f"best_checkpoint_value={best_checkpoint_value:.6f}, "
            f"best_checkpoint_epoch={best_checkpoint_epoch}"
        )

    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Basemodel Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.basemodel_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    logger.info("Loading pretrained model or random initialization...")
    print("Loading pretrained model or random initialization...")
    if getattr(config, 'pre_trained_tokenizer', True):
        tokenizer_load_path = config.finetuned_tokenizer_path
        if (
            (not getattr(config, 'train_tokenizer', True))
            and isinstance(tokenizer_load_path, str)
            and ("/" in tokenizer_load_path or tokenizer_load_path.startswith("."))
            and (not os.path.exists(tokenizer_load_path))
        ):
            tokenizer_load_path = config.pretrained_tokenizer_path
        logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_load_path)
    else:
        import json
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture for training")
        cfg_path_tok = os.path.join(config.pretrained_tokenizer_path if hasattr(config, 'pretrained_tokenizer_path') else config.finetuned_tokenizer_path, 'config.json')
        with open(cfg_path_tok, 'r') as f:
            arch_t = json.load(f)
        tokenizer = KronosTokenizer(
            d_in=arch_t.get('d_in', 6),
            d_model=arch_t.get('d_model', 256),
            n_heads=arch_t.get('n_heads', 4),
            ff_dim=arch_t.get('ff_dim', 512),
            n_enc_layers=arch_t.get('n_enc_layers', 4),
            n_dec_layers=arch_t.get('n_dec_layers', 4),
            ffn_dropout_p=arch_t.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch_t.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch_t.get('resid_dropout_p', 0.0),
            s1_bits=arch_t.get('s1_bits', 10),
            s2_bits=arch_t.get('s2_bits', 10),
            beta=arch_t.get('beta', 0.05),
            gamma0=arch_t.get('gamma0', 1.0),
            gamma=arch_t.get('gamma', 1.1),
            zeta=arch_t.get('zeta', 0.05),
            group_size=arch_t.get('group_size', 4)
        )

    if getattr(config, 'pre_trained_predictor', True):
        model = Kronos.from_pretrained(
            config.pretrained_predictor_path,
            use_output_adapter=getattr(config, 'predictor_use_output_adapter', False),
            output_adapter_bias=getattr(config, 'predictor_output_adapter_bias', True),
            output_adapter_residual=getattr(config, 'predictor_output_adapter_residual', True),
        )
    else:
        import json
        print("pre_trained_predictor=False, randomly initializing Predictor architecture for training")
        cfg_path = os.path.join(config.pretrained_predictor_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        model = Kronos(
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            n_layers=arch.get('n_layers', 12),
            d_model=arch.get('d_model', 832),
            n_heads=arch.get('n_heads', 16),
            ff_dim=arch.get('ff_dim', 2048),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.2),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.2),
            token_dropout_p=arch.get('token_dropout_p', 0.0),
            learn_te=arch.get('learn_te', True),
            use_output_adapter=getattr(config, 'predictor_use_output_adapter', False),
            output_adapter_bias=getattr(config, 'predictor_output_adapter_bias', True),
            output_adapter_residual=getattr(config, 'predictor_output_adapter_residual', True),
        )

    _apply_predictor_dropout_overrides(model, config, logger)
    
    pretrained_reference_state = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }

    tokenizer = tokenizer.to(device)
    model = model.to(device)
    
    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {model_size:,}")
    print(f"Model parameters: {model_size:,}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.predictor_learning_rate}")
    logger.info(f"Training epochs: {config.basemodel_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Tokenizer path: {config.finetuned_tokenizer_path}")
    logger.info(f"Pretrained model path: {config.pretrained_predictor_path}")
    
    logger.info("Starting fine-tuning training...")
    print("Starting fine-tuning training...")
    best_val_loss = train_model(
        model,
        tokenizer,
        device,
        config,
        config.basemodel_save_path,
        logger,
        pretrained_reference_state=pretrained_reference_state,
    )
    
    final_msg = f"Training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.basemodel_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
