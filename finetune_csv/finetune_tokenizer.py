import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import datetime
import logging
from logging.handlers import RotatingFileHandler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("../")
from model import KronosTokenizer
from finetune_base_model import CustomKlineDataset
from config_loader import CustomFinetuneConfig
from training_charts import save_training_dashboard


PRICE_CLOSE_COL = 3


def set_seed(seed: int, rank: int = 0):
    actual_seed = seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model_size(model: torch.nn.Module) -> str:
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params >= 1e9:
        return f"{total_params / 1e9:.1f}B"
    elif total_params >= 1e6:
        return f"{total_params / 1e6:.1f}M"
    else:
        return f"{total_params / 1e3:.1f}K"


def format_time(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds)))


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"tokenizer_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"tokenizer_training_rank_{rank}.log")
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
    
    logger.info(f"=== Tokenizer Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating tokenizer training data loaders...")

    train_data_path = getattr(config, "train_data_path", None) or config.data_path
    val_data_path = getattr(config, "val_data_path", None) or config.data_path
    train_data_mode = str(getattr(config, "train_data_mode", "train")).strip().lower()
    val_data_mode = str(getattr(config, "val_data_mode", "val")).strip().lower()
    
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
        sampling_strategy=getattr(config, "train_sampling_strategy", "independent"),
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
        sampling_strategy=getattr(config, "val_sampling_strategy", "consecutive"),
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
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
        print(f"Tokenizer training data source: {train_data_path} ({train_data_mode})")
        print(f"Tokenizer validation data source: {val_data_path} ({val_data_mode})")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


def directional_loss_and_accuracy(pred_seq: torch.Tensor, target_seq: torch.Tensor):
    """
    Compute directional loss/accuracy on close-price movement.

    pred_seq and target_seq are expected to be [B, T, F] with close at index PRICE_CLOSE_COL.
    """
    if pred_seq.dim() != 3 or target_seq.dim() != 3:
        raise ValueError("pred_seq and target_seq must be rank-3 tensors [B, T, F].")
    if pred_seq.shape != target_seq.shape:
        raise ValueError(f"Shape mismatch: pred_seq={pred_seq.shape}, target_seq={target_seq.shape}")
    if pred_seq.size(1) < 2:
        raise ValueError("Need at least 2 timesteps to compute directional movement.")

    pred_close = pred_seq[:, :, PRICE_CLOSE_COL]
    target_close = target_seq[:, :, PRICE_CLOSE_COL]

    pred_delta = pred_close[:, 1:] - pred_close[:, :-1]
    target_delta = target_close[:, 1:] - target_close[:, :-1]

    target_up = (target_delta >= 0).float()
    dir_loss = F.binary_cross_entropy_with_logits(pred_delta, target_up)
    dir_acc = ((pred_delta >= 0) == (target_up > 0.5)).float().mean()
    return dir_loss, dir_acc


def train_tokenizer(model, device, config, save_dir, logger):
    logger.info("Starting tokenizer training...")
    logger.info("Tokenizer objective: directional close-move BCEWithLogits + BSQ regularization")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tokenizer_learning_rate,
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.tokenizer_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float("inf")
    batch_idx_global = 0
    train_start_time = time.time()
    step_rows: list[dict[str, float]] = []
    epoch_rows: list[dict[str, float]] = []
    
    accumulation_steps = getattr(config, 'accumulation_steps', 1)
    
    for epoch in range(config.tokenizer_epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_train_loss_sum = 0.0
        epoch_train_bsq_sum = 0.0
        epoch_train_dir_loss_pre_sum = 0.0
        epoch_train_dir_loss_all_sum = 0.0
        epoch_train_dir_acc_pre_sum = 0.0
        epoch_train_dir_acc_all_sum = 0.0
        epoch_train_batches = 0
        
        for batch_idx, (ori_batch_x, _) in enumerate(train_loader):
            ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
            
            current_batch_total_loss = 0.0
            current_batch_dir_loss_pre = 0.0
            current_batch_dir_loss_all = 0.0
            current_batch_dir_acc_pre = 0.0
            current_batch_dir_acc_all = 0.0
            current_batch_bsq = 0.0
            for j in range(accumulation_steps):
                start_idx = j * (ori_batch_x.shape[0] // accumulation_steps)
                end_idx = (j + 1) * (ori_batch_x.shape[0] // accumulation_steps)
                batch_x = ori_batch_x[start_idx:end_idx]
                
                zs, bsq_loss, _, _ = (model.module if use_ddp else model)(batch_x)
                z_pre, z = zs

                dir_loss_pre, dir_acc_pre = directional_loss_and_accuracy(z_pre, batch_x)
                dir_loss_all, dir_acc_all = directional_loss_and_accuracy(z, batch_x)
                direction_loss = (dir_loss_pre + dir_loss_all) / 2
                loss = (direction_loss + bsq_loss) / 2
                
                loss_scaled = loss / accumulation_steps
                current_batch_total_loss += loss.item()
                current_batch_dir_loss_pre += dir_loss_pre.item()
                current_batch_dir_loss_all += dir_loss_all.item()
                current_batch_dir_acc_pre += dir_acc_pre.item()
                current_batch_dir_acc_all += dir_acc_all.item()
                current_batch_bsq += bsq_loss.item()
                loss_scaled.backward()
            
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            avg_loss = current_batch_total_loss / accumulation_steps
            avg_dir_loss_pre = current_batch_dir_loss_pre / accumulation_steps
            avg_dir_loss_all = current_batch_dir_loss_all / accumulation_steps
            avg_dir_acc_pre = current_batch_dir_acc_pre / accumulation_steps
            avg_dir_acc_all = current_batch_dir_acc_all / accumulation_steps
            avg_bsq = current_batch_bsq / accumulation_steps

            epoch_train_loss_sum += avg_loss
            epoch_train_bsq_sum += avg_bsq
            epoch_train_dir_loss_pre_sum += avg_dir_loss_pre
            epoch_train_dir_loss_all_sum += avg_dir_loss_all
            epoch_train_dir_acc_pre_sum += avg_dir_acc_pre
            epoch_train_dir_acc_all_sum += avg_dir_acc_all
            epoch_train_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                log_msg = (f"[Epoch {epoch+1}/{config.tokenizer_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {avg_loss:.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
                
                detail_msg = (
                    f"  - VQ Loss: {avg_bsq:.4f}\n"
                    f"  - Dir Loss Pre: {avg_dir_loss_pre:.4f}\n"
                    f"  - Dir Loss All: {avg_dir_loss_all:.4f}\n"
                    f"  - Dir Acc Pre: {avg_dir_acc_pre*100.0:.2f}%\n"
                    f"  - Dir Acc All: {avg_dir_acc_all*100.0:.2f}%"
                )
                logger.info(detail_msg)
                if rank == 0:
                    print(detail_msg)
                    step_rows.append(
                        {
                            "epoch": float(epoch + 1),
                            "step_in_epoch": float(batch_idx + 1),
                            "global_step": float(batch_idx_global + 1),
                            "lr": float(lr),
                            "loss_total": float(avg_loss),
                            "bsq_loss": float(avg_bsq),
                            "dir_loss_pre": float(avg_dir_loss_pre),
                            "dir_loss_all": float(avg_dir_loss_all),
                            "dir_acc_pre_pct": float(avg_dir_acc_pre * 100.0),
                            "dir_acc_all_pct": float(avg_dir_acc_all * 100.0),
                        }
                    )
            
            batch_idx_global += 1
        
        model.eval()
        tot_val_loss_sum_rank = 0.0
        tot_val_dir_acc_sum_rank = 0.0
        val_sample_count_rank = 0
        
        with torch.no_grad():
            for ori_batch_x, _ in val_loader:
                ori_batch_x = ori_batch_x.squeeze(0).to(device, non_blocking=True)
                zs, bsq_loss, _, _ = (model.module if use_ddp else model)(ori_batch_x)
                z_pre, z = zs
                val_dir_loss_pre, val_dir_acc_pre = directional_loss_and_accuracy(z_pre, ori_batch_x)
                val_dir_loss_all, val_dir_acc_all = directional_loss_and_accuracy(z, ori_batch_x)
                val_dir_loss = (val_dir_loss_pre + val_dir_loss_all) / 2
                val_dir_acc = (val_dir_acc_pre + val_dir_acc_all) / 2
                val_loss_item = (val_dir_loss + bsq_loss) / 2
                
                tot_val_loss_sum_rank += val_loss_item.item() * ori_batch_x.size(0)
                tot_val_dir_acc_sum_rank += val_dir_acc.item() * ori_batch_x.size(0)
                val_sample_count_rank += ori_batch_x.size(0)
        
        if use_ddp:
            tensor_sum = torch.tensor(
                [tot_val_loss_sum_rank, tot_val_dir_acc_sum_rank, val_sample_count_rank],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            tot_val_loss_all = tensor_sum[0].item()
            tot_val_dir_acc_all = tensor_sum[1].item()
            val_count_all = int(tensor_sum[2].item())
            avg_val_loss = (tot_val_loss_all / val_count_all) if val_count_all > 0 else 0.0
            avg_val_dir_acc = (tot_val_dir_acc_all / val_count_all) if val_count_all > 0 else 0.0
        else:
            avg_val_loss = tot_val_loss_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0
            avg_val_dir_acc = tot_val_dir_acc_sum_rank / val_sample_count_rank if val_sample_count_rank > 0 else 0
        avg_train_loss = epoch_train_loss_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        avg_train_bsq = epoch_train_bsq_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        avg_train_dir_loss_pre = epoch_train_dir_loss_pre_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        avg_train_dir_loss_all = epoch_train_dir_loss_all_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        avg_train_dir_acc_pre = epoch_train_dir_acc_pre_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        avg_train_dir_acc_all = epoch_train_dir_acc_all_sum / epoch_train_batches if epoch_train_batches > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.tokenizer_epochs} Summary ---\n"
                       f"Train Loss: {avg_train_loss:.4f}\n"
                       f"Train VQ Loss: {avg_train_bsq:.4f}\n"
                       f"Train Dir Loss Pre: {avg_train_dir_loss_pre:.4f}\n"
                       f"Train Dir Loss All: {avg_train_dir_loss_all:.4f}\n"
                       f"Train Dir Acc Pre: {avg_train_dir_acc_pre*100.0:.2f}%\n"
                       f"Train Dir Acc All: {avg_train_dir_acc_all*100.0:.2f}%\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Validation Directional Accuracy: {avg_val_dir_acc*100.0:.2f}%\n"
                       f"Epoch Time: {format_time(epoch_time)}\n"
                       f"Total Training Time: {format_time(time.time() - train_start_time)}\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
            epoch_rows.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": float(avg_train_loss),
                    "train_bsq_loss": float(avg_train_bsq),
                    "train_dir_loss_pre": float(avg_train_dir_loss_pre),
                    "train_dir_loss_all": float(avg_train_dir_loss_all),
                    "train_dir_acc_pre_pct": float(avg_train_dir_acc_pre * 100.0),
                    "train_dir_acc_all_pct": float(avg_train_dir_acc_all * 100.0),
                    "val_loss": float(avg_val_loss),
                    "val_dir_acc_pct": float(avg_val_dir_acc * 100.0),
                    "epoch_time_sec": float(epoch_time),
                    "elapsed_total_sec": float(time.time() - train_start_time),
                    "best_val_loss_so_far": float(min(best_val_loss, avg_val_loss)),
                }
            )
            metrics_dir = getattr(config, "metrics_dir", os.path.join(config.base_save_path, "metrics"))
            save_training_dashboard(
                out_dir=metrics_dir,
                prefix="tokenizer",
                step_rows=step_rows,
                epoch_rows=epoch_rows,
                logger=logger,
            )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
    if rank == 0:
        metrics_dir = getattr(config, "metrics_dir", os.path.join(config.base_save_path, "metrics"))
        dashboard = save_training_dashboard(
            out_dir=metrics_dir,
            prefix="tokenizer",
            step_rows=step_rows,
            epoch_rows=epoch_rows,
            logger=logger,
        )
        logger.info(f"Tokenizer charts and metrics saved: {json.dumps(dashboard, indent=2)}")

    return best_val_loss


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kronos Tokenizer Fine-tuning Training')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Configuration file path (default: config.yaml)')
    args = parser.parse_args()
    
    config = CustomFinetuneConfig(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = CustomFinetuneConfig(args.config)
    
    os.makedirs(config.tokenizer_save_path, exist_ok=True)
    
    log_dir = os.path.join(config.base_save_path, "logs")
    logger = setup_logging(config.exp_name, log_dir, 0)
    
    set_seed(config.seed)
    
    # 加载预训练tokenizer
    if getattr(config, 'pre_trained_tokenizer', True):
        logger.info("Loading pretrained tokenizer...")
        print("Loading pretrained tokenizer...")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    else:
        print("pre_trained_tokenizer=False, randomly initializing Tokenizer architecture")
        import json
        cfg_path = os.path.join(config.pretrained_tokenizer_path, 'config.json')
        with open(cfg_path, 'r') as f:
            arch = json.load(f)
        tokenizer = KronosTokenizer(
            d_in=arch.get('d_in', 6),
            d_model=arch.get('d_model', 256),
            n_heads=arch.get('n_heads', 4),
            ff_dim=arch.get('ff_dim', 512),
            n_enc_layers=arch.get('n_enc_layers', 4),
            n_dec_layers=arch.get('n_dec_layers', 4),
            ffn_dropout_p=arch.get('ffn_dropout_p', 0.0),
            attn_dropout_p=arch.get('attn_dropout_p', 0.0),
            resid_dropout_p=arch.get('resid_dropout_p', 0.0),
            s1_bits=arch.get('s1_bits', 10),
            s2_bits=arch.get('s2_bits', 10),
            beta=arch.get('beta', 0.05),
            gamma0=arch.get('gamma0', 1.0),
            gamma=arch.get('gamma', 1.1),
            zeta=arch.get('zeta', 0.05),
            group_size=arch.get('group_size', 4)
        )
    tokenizer = tokenizer.to(device)
    
    model_size = get_model_size(tokenizer)
    logger.info(f"Tokenizer parameters: {model_size}")
    print(f"Tokenizer parameters: {model_size}")
    
    logger.info("=== Training Configuration ===")
    logger.info(f"Data path: {config.data_path}")
    logger.info(f"Lookback window: {config.lookback_window}")
    logger.info(f"Predict window: {config.predict_window}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.tokenizer_learning_rate}")
    logger.info(f"Training epochs: {config.tokenizer_epochs}")
    logger.info(f"Device: {device}")
    logger.info(f"Distributed training: False")
    
    logger.info("Starting tokenizer fine-tuning training...")
    print("Starting tokenizer fine-tuning training...")
    best_val_loss = train_tokenizer(tokenizer, device, config, config.tokenizer_save_path, logger)
    
    final_msg = f"Tokenizer training completed! Best validation loss: {best_val_loss:.4f}\nModel saved to: {config.tokenizer_save_path}"
    logger.info(final_msg)
    print(final_msg)


if __name__ == "__main__":
    main()
    
