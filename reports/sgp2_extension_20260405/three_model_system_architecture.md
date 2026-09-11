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

## Public artifact references

- [5-minute native direction-head checkpoint](https://huggingface.co/abdelrahman964/kronos-finetuning/blob/main/models/5min-native-direction-head/best_direction_model.pt)
- [1-hour 50% pruned direction-head checkpoint](https://huggingface.co/abdelrahman964/kronos-finetuning/blob/main/models/1h-pruned50-direction-head/best_direction_model.pt)
- [1-day last-six-layer path-adapter checkpoint](https://huggingface.co/abdelrahman964/kronos-finetuning/blob/main/models/1d-last6-path-adapter/model.safetensors)
- [Fine-tuned 5-minute Kronos predictor](https://huggingface.co/abdelrahman964/kronos-finetuning/tree/main/models/5min-kronos-base)
- [Fine-tuned 5-minute tokenizer](https://huggingface.co/abdelrahman964/kronos-finetuning/tree/main/models/5min-tokenizer)

The complete machine-readable inventory is available in the Hugging Face
[`weights_manifest.json`](https://huggingface.co/abdelrahman964/kronos-finetuning/blob/main/weights_manifest.json).
