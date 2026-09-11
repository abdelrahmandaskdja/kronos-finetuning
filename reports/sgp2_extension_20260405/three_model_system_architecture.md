# Three-Model System Architecture

This diagram captures the current recommended 2026 deployment stack for the three selected BTCUSDT models:

- `5min`: native direction head
- `1h`: pruned-50 direction head
- `1d`: last-6 consecutive-path adapter




## Plain-Text View

```text
BTCUSDT candles
  -> interval loader (5min / 1h / 1d)
  -> shared preprocessing
  -> timeframe router

Router outputs:
  1. 5min route
     -> 5min native direction-head checkpoint
     -> one-step up/down signal

  2. 1h route
     -> 1h pruned50 direction-head checkpoint
     -> one-step up/down signal

  3. 1d route
     -> 1d last6 consecutive-path adapter checkpoint
     -> multi-step path rollout
     -> next-day direction extracted from rollout

All three routes then feed:
  -> signal standardization layer
  -> backtesting / trading / dashboard / monitoring
```

## Model Roles

- `5min native direction head`
  Best saved short-horizon native model for `5min`.
- `1h pruned50 direction head`
  Best `1h` deployment checkpoint and cleanest pruning-based winner.
- `1d last6 consecutive-path adapter`
  Best saved daily deployment result, but more sensitive to later-window drift.

## Artifact References

- `5min checkpoint`
  `./finetune_csv/runs_requested_by_user/research_5min_direction_head_oldsplit_20260318/checkpoints/best_direction_model.pt`
- `1h checkpoint`
  `./finetune_csv/direction_head_ce_frozen_5min2018_2025_20260304_gpu_pruned50/best_direction_model.pt`
- `1d checkpoint`
  `./finetune_csv/runs_requested_by_user/research_1m_multistep_from_epoch4_20260311/screening_trials/cons_only_lr3e6_last6_adapter/finetuned/cons_only_lr3e6_last6_adapter/basemodel/best_model`
