# Reproducibility guide

This guide separates the saved research evidence from claims that require a
fresh run. The repository contains evaluation tables, figures, raw predictions,
and model-selection manifests. Model binaries are hosted on Hugging Face.

## 1. Environment

- Python 3.10 or newer
- PyTorch with a CUDA build for GPU evaluation, or CPU for smaller checks
- Dependencies from `requirements.txt`
- Seed `123` for `finetune_csv/eval_direction_model.py` unless an experiment
  explicitly records another value

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install huggingface_hub
```

## 2. Download the public checkpoints

```bash
huggingface-cli download abdelrahman964/kronos-finetuning \
  --include "models/**" "weights_manifest.json" \
  --local-dir artifacts/kronos-finetuning
```

Verify downloaded files against `weights_manifest.json` or the checksums in
`MODEL_WEIGHTS.md`.

## 3. Recreate market data

The downloader uses Binance's public kline endpoints and does not require an
API key. Declare the exact UTC range so a later rerun does not silently use a
different window.

```bash
python finetune_csv/download_klines.py \
  --symbol BTCUSDT \
  --interval 5m \
  --start 2018-01-01 \
  --end 2026-01-01 \
  --output finetune_csv/data/BTCUSDT_kline_5min_2018_2025.csv
```

Repeat with `--interval 1h` or `--interval 1m` for the associated experiment.
The daily path-adapter experiment derives daily targets from the recorded
1-minute data pipeline.

## 4. Evaluate direction-head checkpoints

The direction-head evaluator accepts explicit paths so stale training-machine
paths embedded in checkpoint metadata do not have to exist.

```bash
python finetune_csv/eval_direction_model.py \
  --data-path finetune_csv/data/BTCUSDT_kline_5min_2018_2025.csv \
  --checkpoint-path artifacts/kronos-finetuning/models/5min-native-direction-head/best_direction_model.pt \
  --predictor-path artifacts/kronos-finetuning/models/5min-kronos-base \
  --tokenizer-path artifacts/kronos-finetuning/models/5min-tokenizer \
  --data-type test \
  --seed 123 \
  --output-json artifacts/evaluation/5min_metrics.json \
  --output-csv artifacts/evaluation/5min_predictions.csv
```

For the 1-hour pruned checkpoint, replace `--checkpoint-path` with
`models/1h-pruned50-direction-head/best_direction_model.pt` and use the exact
hourly dataset and split recorded by that experiment.

## 5. Audit the saved results

The principal summary tables are:

- `reports/sgp2_extension_20260405/selected_winners_summary.csv`
- `reports/sgp2_extension_20260405/hourly_pruning_sweep.csv`
- `reports/sgp2_extension_20260405/daily_screening_summary.csv`
- `reports/sgp2_extension_20260405/daily_winner_extension.csv`
- `reports/ad_hoc_20260409/cross_timeframe_direction_heads_2026-04-08/`

The headline values use different evaluation windows and sample counts. Do not
treat them as a single controlled cross-horizon benchmark. The later daily
extension is retained beside the best snapshot to show the observed reduction
in generalization margin.

## 6. Backtest assumptions

Reported PnL assumes a fixed 1 BTC position and frictionless execution. It omits
fees, bid-ask spread, slippage, latency, liquidity limits, funding, and market
impact. Reproducing the arithmetic does not establish deployable profitability.
