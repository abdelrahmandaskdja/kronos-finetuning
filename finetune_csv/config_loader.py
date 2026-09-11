import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    
    def __init__(self, config_path: str):

        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config = self._resolve_dynamic_paths(config)
        
        return config
    
    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:

        exp_name = config.get('model_paths', {}).get('exp_name', '')
        if not exp_name:
            return config
        
        base_path = config.get('model_paths', {}).get('base_path', '')
        path_templates = {
            'base_save_path': f"{base_path}/{exp_name}",
            'finetuned_tokenizer': f"{base_path}/{exp_name}/tokenizer/best_model"
        }
        
        if 'model_paths' in config:
            for key, template in path_templates.items():
                if key in config['model_paths']:
                    # only use template when the original value is empty string
                    current_value = config['model_paths'][key]
                    if current_value == "" or current_value is None:
                        config['model_paths'][key] = template
                    else:
                        # if the original value is not empty, use template to replace the {exp_name} placeholder
                        if isinstance(current_value, str) and '{exp_name}' in current_value:
                            config['model_paths'][key] = current_value.format(exp_name=exp_name)
        
        return config
    
    def get(self, key: str, default=None):
 
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_model_paths(self) -> Dict[str, str]:
        return self.config.get('model_paths', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})
    
    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get('device', {})
    
    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})
    
    def update_config(self, updates: Dict[str, Any]):

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
    
    def save_config(self, save_path: str = None):

        if save_path is None:
            save_path = self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:
    
    def __init__(self, config_path: str = None):

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.loader = ConfigLoader(config_path)
        self._load_all_configs()
    
    def _load_all_configs(self):

        data_config = self.loader.get_data_config()
        self.data_path = data_config.get('data_path')
        self.train_data_path = data_config.get('train_data_path', self.data_path)
        self.val_data_path = data_config.get('val_data_path', self.data_path)
        self.train_data_mode = data_config.get('train_data_mode', 'train')
        self.val_data_mode = data_config.get('val_data_mode', 'val')
        self.lookback_window = data_config.get('lookback_window', 512)
        self.predict_window = data_config.get('predict_window', 48)
        self.max_context = data_config.get('max_context', 512)
        self.clip = data_config.get('clip', 5.0)
        self.train_ratio = data_config.get('train_ratio', 0.9)
        self.val_ratio = data_config.get('val_ratio', 0.1)
        self.test_ratio = data_config.get('test_ratio', 0.0)
        
        # training configuration
        training_config = self.loader.get_training_config()
        # support training epochs of tokenizer and basemodel separately
        self.tokenizer_epochs = training_config.get('tokenizer_epochs', 30)
        self.basemodel_epochs = training_config.get('basemodel_epochs', 30)

        if 'epochs' in training_config and 'tokenizer_epochs' not in training_config:
            self.tokenizer_epochs = training_config.get('epochs', 30)
        if 'epochs' in training_config and 'basemodel_epochs' not in training_config:
            self.basemodel_epochs = training_config.get('epochs', 30)
        
        self.batch_size = training_config.get('batch_size', 160)
        self.log_interval = training_config.get('log_interval', 50)
        self.num_workers = training_config.get('num_workers', 6)
        self.seed = training_config.get('seed', 100)
        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)
        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)
        self.accumulation_steps = training_config.get('accumulation_steps', 1)
        self.predictor_ce_loss_weight = training_config.get('predictor_ce_loss_weight', 1.0)
        self.predictor_directional_loss_weight = training_config.get('predictor_directional_loss_weight', 0.0)
        self.predictor_consecutive_directional_loss_weight = training_config.get('predictor_consecutive_directional_loss_weight', 0.0)
        self.predictor_consecutive_pos_weight = training_config.get('predictor_consecutive_pos_weight', 1.0)
        self.predictor_consecutive_focal_gamma = training_config.get('predictor_consecutive_focal_gamma', 0.0)
        self.predictor_consecutive_label_smoothing = training_config.get('predictor_consecutive_label_smoothing', 0.0)
        self.predictor_consecutive_negative_weight = training_config.get(
            'predictor_consecutive_negative_weight',
            1.0,
        )
        self.predictor_consecutive_false_positive_weight = training_config.get(
            'predictor_consecutive_false_positive_weight',
            0.0,
        )
        self.predictor_consecutive_bias_penalty_weight = training_config.get(
            'predictor_consecutive_bias_penalty_weight',
            0.0,
        )
        self.predictor_consecutive_magnitude_mode = training_config.get(
            'predictor_consecutive_magnitude_mode',
            'relative_return',
        )
        self.predictor_consecutive_magnitude_weight = training_config.get('predictor_consecutive_magnitude_weight', 0.0)
        self.predictor_consecutive_magnitude_scale = training_config.get('predictor_consecutive_magnitude_scale', 0.01)
        self.predictor_consecutive_magnitude_power = training_config.get('predictor_consecutive_magnitude_power', 1.0)
        self.predictor_consecutive_magnitude_cap = training_config.get('predictor_consecutive_magnitude_cap', 5.0)
        self.predictor_mse_loss_weight = training_config.get('predictor_mse_loss_weight', 0.0)
        self.predictor_objective_mode = training_config.get('predictor_objective_mode', 'hybrid')
        self.predictor_use_output_adapter = training_config.get('predictor_use_output_adapter', False)
        self.predictor_output_adapter_bias = training_config.get('predictor_output_adapter_bias', True)
        self.predictor_output_adapter_residual = training_config.get('predictor_output_adapter_residual', True)
        self.predictor_train_last_n_transformer_layers = training_config.get('predictor_train_last_n_transformer_layers', -1)
        self.predictor_override_ffn_dropout_p = training_config.get('predictor_override_ffn_dropout_p', None)
        self.predictor_override_attn_dropout_p = training_config.get('predictor_override_attn_dropout_p', None)
        self.predictor_override_resid_dropout_p = training_config.get('predictor_override_resid_dropout_p', None)
        self.predictor_override_token_dropout_p = training_config.get('predictor_override_token_dropout_p', None)
        self.train_sampling_strategy = training_config.get('train_sampling_strategy', 'independent')
        self.val_sampling_strategy = training_config.get('val_sampling_strategy', 'consecutive')
        self.metrics_dir = training_config.get('metrics_dir', '')
        self.predictor_checkpoint_metric = training_config.get('predictor_checkpoint_metric', 'val_loss')
        self.predictor_rollout_metric_key = training_config.get(
            'predictor_rollout_metric_key',
            'directional_accuracy_close_vs_prev_close_consecutive_path',
        )
        self.predictor_rollout_train_context_path = training_config.get('predictor_rollout_train_context_path', '')
        self.predictor_rollout_eval_path = training_config.get('predictor_rollout_eval_path', '')
        self.predictor_rollout_lookback = training_config.get('predictor_rollout_lookback', 0)
        self.predictor_rollout_block_len = training_config.get('predictor_rollout_block_len', 0)
        self.predictor_rollout_feedback_source = training_config.get('predictor_rollout_feedback_source', 'actual')
        self.predictor_rollout_device = training_config.get('predictor_rollout_device', '')
        self.predictor_rollout_trade_fee_bps = training_config.get('predictor_rollout_trade_fee_bps', 0.0)
        self.predictor_rollout_eval_interval_epochs = training_config.get('predictor_rollout_eval_interval_epochs', 1)
        self.predictor_rollout_stabilize_output = training_config.get('predictor_rollout_stabilize_output', True)
        self.predictor_rollout_stability_apply_to_context = training_config.get(
            'predictor_rollout_stability_apply_to_context',
            False,
        )
        self.predictor_scheduled_sampling_max_rate = training_config.get('predictor_scheduled_sampling_max_rate', 0.0)
        self.predictor_scheduled_sampling_warmup_epochs = training_config.get(
            'predictor_scheduled_sampling_warmup_epochs',
            0,
        )
        self.predictor_scheduled_sampling_ramp_epochs = training_config.get(
            'predictor_scheduled_sampling_ramp_epochs',
            0,
        )
        self.predictor_scheduled_sampling_strategy = training_config.get(
            'predictor_scheduled_sampling_strategy',
            'tail',
        )
        self.predictor_scheduled_sampling_mode = training_config.get(
            'predictor_scheduled_sampling_mode',
            'greedy',
        )
        self.predictor_scheduler_name = training_config.get('predictor_scheduler_name', 'onecycle')
        self.predictor_scheduler_warmup_steps = training_config.get('predictor_scheduler_warmup_steps', 0)
        self.predictor_scheduler_warmup_ratio = training_config.get('predictor_scheduler_warmup_ratio', 0.0)
        self.predictor_scheduler_warmup_start_factor = training_config.get(
            'predictor_scheduler_warmup_start_factor',
            0.1,
        )
        self.predictor_scheduler_min_lr_ratio = training_config.get('predictor_scheduler_min_lr_ratio', 0.0)
        self.predictor_scheduler_plateau_factor = training_config.get('predictor_scheduler_plateau_factor', 0.5)
        self.predictor_scheduler_plateau_patience = training_config.get(
            'predictor_scheduler_plateau_patience',
            2,
        )
        self.predictor_scheduler_plateau_metric = training_config.get(
            'predictor_scheduler_plateau_metric',
            'checkpoint_metric',
        )
        self.predictor_scheduler_restart_t0 = training_config.get('predictor_scheduler_restart_t0', 0)
        self.predictor_scheduler_restart_t_mult = training_config.get(
            'predictor_scheduler_restart_t_mult',
            1,
        )
        self.predictor_scheduler_onecycle_pct_start = training_config.get(
            'predictor_scheduler_onecycle_pct_start',
            0.03,
        )
        self.predictor_scheduler_onecycle_div_factor = training_config.get(
            'predictor_scheduler_onecycle_div_factor',
            10.0,
        )
        self.predictor_scheduler_onecycle_final_div_factor = training_config.get(
            'predictor_scheduler_onecycle_final_div_factor',
            100.0,
        )
        self.predictor_layerwise_lr_decay = training_config.get('predictor_layerwise_lr_decay', 1.0)
        self.predictor_weight_decay_exclude_bias_norm = training_config.get(
            'predictor_weight_decay_exclude_bias_norm',
            False,
        )
        self.predictor_pretrained_anchor_weight = training_config.get(
            'predictor_pretrained_anchor_weight',
            0.0,
        )
        self.predictor_ema_decay = training_config.get('predictor_ema_decay', 0.0)
        self.predictor_ema_start_step = training_config.get('predictor_ema_start_step', 0)
        self.predictor_ema_eval = training_config.get('predictor_ema_eval', False)
        self.predictor_early_stopping_patience = training_config.get(
            'predictor_early_stopping_patience',
            0,
        )
        self.predictor_early_stopping_min_delta = training_config.get(
            'predictor_early_stopping_min_delta',
            0.0,
        )
        
        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get('exp_name', 'default_experiment')
        self.pretrained_tokenizer_path = model_paths.get('pretrained_tokenizer')
        self.pretrained_predictor_path = model_paths.get('pretrained_predictor')
        self.base_save_path = model_paths.get('base_save_path')
        self.tokenizer_save_name = model_paths.get('tokenizer_save_name', 'tokenizer')
        self.basemodel_save_name = model_paths.get('basemodel_save_name', 'basemodel')
        self.finetuned_tokenizer_path = model_paths.get('finetuned_tokenizer')
        
        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get('name', 'kronos_custom_finetune')
        self.experiment_description = experiment_config.get('description', '')
        self.use_comet = experiment_config.get('use_comet', False)
        self.train_tokenizer = experiment_config.get('train_tokenizer', True)
        self.train_basemodel = experiment_config.get('train_basemodel', True)
        self.skip_existing = experiment_config.get('skip_existing', False)

        unified_pretrained = experiment_config.get('pre_trained', None)
        self.pre_trained_tokenizer = experiment_config.get('pre_trained_tokenizer', unified_pretrained if unified_pretrained is not None else True)
        self.pre_trained_predictor = experiment_config.get('pre_trained_predictor', unified_pretrained if unified_pretrained is not None else True)
        
        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get('use_cuda', True)
        self.device_id = device_config.get('device_id', 0)
        
        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get('use_ddp', False)
        self.ddp_backend = distributed_config.get('backend', 'nccl')
        
        self._compute_full_paths()
    
    def _compute_full_paths(self):

        self.tokenizer_save_path = os.path.join(self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(self.tokenizer_save_path, 'best_model')
        
        self.basemodel_save_path = os.path.join(self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(self.basemodel_save_path, 'best_model')
        if not self.metrics_dir:
            self.metrics_dir = os.path.join(self.base_save_path, 'metrics')
        if (not self.train_tokenizer) and (
            not self.finetuned_tokenizer_path or not os.path.exists(self.finetuned_tokenizer_path)
        ):
            self.finetuned_tokenizer_path = self.pretrained_tokenizer_path
    
    def get_tokenizer_config(self):

        return {
            'data_path': self.data_path,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'train_data_mode': self.train_data_mode,
            'val_data_mode': self.val_data_mode,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.tokenizer_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'accumulation_steps': self.accumulation_steps,
            'pretrained_model_path': self.pretrained_tokenizer_path,
            'save_path': self.tokenizer_save_path,
            'use_comet': self.use_comet
        }
    
    def get_basemodel_config(self):

        return {
            'data_path': self.data_path,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'train_data_mode': self.train_data_mode,
            'val_data_mode': self.val_data_mode,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.basemodel_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'predictor_learning_rate': self.predictor_learning_rate,
            'tokenizer_learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'predictor_ce_loss_weight': self.predictor_ce_loss_weight,
            'predictor_directional_loss_weight': self.predictor_directional_loss_weight,
            'predictor_consecutive_directional_loss_weight': self.predictor_consecutive_directional_loss_weight,
            'predictor_consecutive_pos_weight': self.predictor_consecutive_pos_weight,
            'predictor_consecutive_focal_gamma': self.predictor_consecutive_focal_gamma,
            'predictor_consecutive_label_smoothing': self.predictor_consecutive_label_smoothing,
            'predictor_consecutive_negative_weight': self.predictor_consecutive_negative_weight,
            'predictor_consecutive_false_positive_weight': self.predictor_consecutive_false_positive_weight,
            'predictor_consecutive_bias_penalty_weight': self.predictor_consecutive_bias_penalty_weight,
            'predictor_consecutive_magnitude_mode': self.predictor_consecutive_magnitude_mode,
            'predictor_consecutive_magnitude_weight': self.predictor_consecutive_magnitude_weight,
            'predictor_consecutive_magnitude_scale': self.predictor_consecutive_magnitude_scale,
            'predictor_consecutive_magnitude_power': self.predictor_consecutive_magnitude_power,
            'predictor_consecutive_magnitude_cap': self.predictor_consecutive_magnitude_cap,
            'predictor_mse_loss_weight': self.predictor_mse_loss_weight,
            'predictor_objective_mode': self.predictor_objective_mode,
            'predictor_use_output_adapter': self.predictor_use_output_adapter,
            'predictor_output_adapter_bias': self.predictor_output_adapter_bias,
            'predictor_output_adapter_residual': self.predictor_output_adapter_residual,
            'predictor_train_last_n_transformer_layers': self.predictor_train_last_n_transformer_layers,
            'predictor_override_ffn_dropout_p': self.predictor_override_ffn_dropout_p,
            'predictor_override_attn_dropout_p': self.predictor_override_attn_dropout_p,
            'predictor_override_resid_dropout_p': self.predictor_override_resid_dropout_p,
            'predictor_override_token_dropout_p': self.predictor_override_token_dropout_p,
            'train_sampling_strategy': self.train_sampling_strategy,
            'val_sampling_strategy': self.val_sampling_strategy,
            'metrics_dir': self.metrics_dir,
            'predictor_checkpoint_metric': self.predictor_checkpoint_metric,
            'predictor_rollout_metric_key': self.predictor_rollout_metric_key,
            'predictor_rollout_train_context_path': self.predictor_rollout_train_context_path,
            'predictor_rollout_eval_path': self.predictor_rollout_eval_path,
            'predictor_rollout_lookback': self.predictor_rollout_lookback,
            'predictor_rollout_block_len': self.predictor_rollout_block_len,
            'predictor_rollout_feedback_source': self.predictor_rollout_feedback_source,
            'predictor_rollout_device': self.predictor_rollout_device,
            'predictor_rollout_trade_fee_bps': self.predictor_rollout_trade_fee_bps,
            'predictor_rollout_eval_interval_epochs': self.predictor_rollout_eval_interval_epochs,
            'predictor_scheduled_sampling_max_rate': self.predictor_scheduled_sampling_max_rate,
            'predictor_scheduled_sampling_warmup_epochs': self.predictor_scheduled_sampling_warmup_epochs,
            'predictor_scheduled_sampling_ramp_epochs': self.predictor_scheduled_sampling_ramp_epochs,
            'predictor_scheduled_sampling_strategy': self.predictor_scheduled_sampling_strategy,
            'predictor_scheduled_sampling_mode': self.predictor_scheduled_sampling_mode,
            'predictor_scheduler_name': self.predictor_scheduler_name,
            'predictor_scheduler_warmup_steps': self.predictor_scheduler_warmup_steps,
            'predictor_scheduler_warmup_ratio': self.predictor_scheduler_warmup_ratio,
            'predictor_scheduler_warmup_start_factor': self.predictor_scheduler_warmup_start_factor,
            'predictor_scheduler_min_lr_ratio': self.predictor_scheduler_min_lr_ratio,
            'predictor_scheduler_plateau_factor': self.predictor_scheduler_plateau_factor,
            'predictor_scheduler_plateau_patience': self.predictor_scheduler_plateau_patience,
            'predictor_scheduler_plateau_metric': self.predictor_scheduler_plateau_metric,
            'predictor_scheduler_restart_t0': self.predictor_scheduler_restart_t0,
            'predictor_scheduler_restart_t_mult': self.predictor_scheduler_restart_t_mult,
            'predictor_scheduler_onecycle_pct_start': self.predictor_scheduler_onecycle_pct_start,
            'predictor_scheduler_onecycle_div_factor': self.predictor_scheduler_onecycle_div_factor,
            'predictor_scheduler_onecycle_final_div_factor': self.predictor_scheduler_onecycle_final_div_factor,
            'pretrained_tokenizer_path': self.finetuned_tokenizer_path,
            'pretrained_predictor_path': self.pretrained_predictor_path,
            'save_path': self.basemodel_save_path,
            'use_comet': self.use_comet
        }
    
    def print_config_summary(self):

        print("=" * 60)
        print("Kronos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name: {self.exp_name}")
        print(f"Data path: {self.data_path}")
        print(f"Train data path: {self.train_data_path} ({self.train_data_mode})")
        print(f"Validation data path: {self.val_data_path} ({self.val_data_mode})")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Predict window: {self.predict_window}")
        print(f"Tokenizer training epochs: {self.tokenizer_epochs}")
        print(f"Basemodel training epochs: {self.basemodel_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tokenizer learning rate: {self.tokenizer_learning_rate}")
        print(f"Predictor learning rate: {self.predictor_learning_rate}")
        print(f"Predictor CE weight: {self.predictor_ce_loss_weight}")
        print(f"Predictor directional weight: {self.predictor_directional_loss_weight}")
        print(f"Predictor consecutive directional weight: {self.predictor_consecutive_directional_loss_weight}")
        print(f"Predictor consecutive pos weight: {self.predictor_consecutive_pos_weight}")
        print(f"Predictor consecutive focal gamma: {self.predictor_consecutive_focal_gamma}")
        print(f"Predictor consecutive label smoothing: {self.predictor_consecutive_label_smoothing}")
        print(f"Predictor consecutive negative weight: {self.predictor_consecutive_negative_weight}")
        print(
            "Predictor consecutive false-positive weight: "
            f"{self.predictor_consecutive_false_positive_weight}"
        )
        print(
            "Predictor consecutive bias penalty weight: "
            f"{self.predictor_consecutive_bias_penalty_weight}"
        )
        print(f"Predictor consecutive magnitude mode: {self.predictor_consecutive_magnitude_mode}")
        print(f"Predictor consecutive magnitude weight: {self.predictor_consecutive_magnitude_weight}")
        print(f"Predictor consecutive magnitude scale: {self.predictor_consecutive_magnitude_scale}")
        print(f"Predictor consecutive magnitude power: {self.predictor_consecutive_magnitude_power}")
        print(f"Predictor consecutive magnitude cap: {self.predictor_consecutive_magnitude_cap}")
        print(f"Predictor MSE weight: {self.predictor_mse_loss_weight}")
        print(f"Predictor objective mode: {self.predictor_objective_mode}")
        print(f"Train sampling strategy: {self.train_sampling_strategy}")
        print(f"Validation sampling strategy: {self.val_sampling_strategy}")
        print(f"Predictor use output adapter: {self.predictor_use_output_adapter}")
        print(f"Predictor output adapter residual: {self.predictor_output_adapter_residual}")
        print(f"Predictor train last N transformer layers: {self.predictor_train_last_n_transformer_layers}")
        print(f"Predictor override FFN dropout: {self.predictor_override_ffn_dropout_p}")
        print(f"Predictor override attention dropout: {self.predictor_override_attn_dropout_p}")
        print(f"Predictor override residual dropout: {self.predictor_override_resid_dropout_p}")
        print(f"Predictor override token dropout: {self.predictor_override_token_dropout_p}")
        print(f"Predictor checkpoint metric: {self.predictor_checkpoint_metric}")
        print(f"Predictor rollout metric key: {self.predictor_rollout_metric_key}")
        print(f"Predictor rollout train context path: {self.predictor_rollout_train_context_path}")
        print(f"Predictor rollout eval path: {self.predictor_rollout_eval_path}")
        print(f"Predictor rollout lookback: {self.predictor_rollout_lookback}")
        print(f"Predictor rollout block len: {self.predictor_rollout_block_len}")
        print(f"Predictor rollout feedback source: {self.predictor_rollout_feedback_source}")
        print(f"Predictor rollout device: {self.predictor_rollout_device}")
        print(f"Predictor rollout trade fee bps: {self.predictor_rollout_trade_fee_bps}")
        print(f"Predictor rollout eval interval epochs: {self.predictor_rollout_eval_interval_epochs}")
        print(f"Predictor scheduled sampling max rate: {self.predictor_scheduled_sampling_max_rate}")
        print(f"Predictor scheduled sampling warmup epochs: {self.predictor_scheduled_sampling_warmup_epochs}")
        print(f"Predictor scheduled sampling ramp epochs: {self.predictor_scheduled_sampling_ramp_epochs}")
        print(f"Predictor scheduled sampling strategy: {self.predictor_scheduled_sampling_strategy}")
        print(f"Predictor scheduled sampling mode: {self.predictor_scheduled_sampling_mode}")
        print(f"Predictor scheduler name: {self.predictor_scheduler_name}")
        print(f"Predictor scheduler warmup steps: {self.predictor_scheduler_warmup_steps}")
        print(f"Predictor scheduler warmup ratio: {self.predictor_scheduler_warmup_ratio}")
        print(
            f"Predictor scheduler warmup start factor: {self.predictor_scheduler_warmup_start_factor}"
        )
        print(f"Predictor scheduler min LR ratio: {self.predictor_scheduler_min_lr_ratio}")
        print(f"Predictor scheduler plateau factor: {self.predictor_scheduler_plateau_factor}")
        print(f"Predictor scheduler plateau patience: {self.predictor_scheduler_plateau_patience}")
        print(f"Predictor scheduler plateau metric: {self.predictor_scheduler_plateau_metric}")
        print(f"Predictor scheduler restart T0: {self.predictor_scheduler_restart_t0}")
        print(f"Predictor scheduler restart T mult: {self.predictor_scheduler_restart_t_mult}")
        print(
            f"Predictor scheduler OneCycle pct_start: {self.predictor_scheduler_onecycle_pct_start}"
        )
        print(
            f"Predictor scheduler OneCycle div factor: {self.predictor_scheduler_onecycle_div_factor}"
        )
        print(
            "Predictor scheduler OneCycle final div factor: "
            f"{self.predictor_scheduler_onecycle_final_div_factor}"
        )
        print(f"Train tokenizer: {self.train_tokenizer}")
        print(f"Train basemodel: {self.train_basemodel}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Use pre-trained tokenizer: {self.pre_trained_tokenizer}")
        print(f"Use pre-trained predictor: {self.pre_trained_predictor}")
        print(f"Base save path: {self.base_save_path}")
        print(f"Tokenizer save path: {self.tokenizer_save_path}")
        print(f"Basemodel save path: {self.basemodel_save_path}")
        print(f"Metrics directory: {self.metrics_dir}")
        print("=" * 60)
