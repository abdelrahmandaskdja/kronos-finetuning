# Kronos Fine-tuning on Custom CSV Datasets

This module provides a comprehensive pipeline for fine-tuning Kronos models on your own CSV-formatted financial data. It supports both sequential training (tokenizer followed by predictor) and individual component training, with full distributed training capabilities.


## 1. Data Preparation

### Required Data Format

Your CSV file must contain the following columns:
- `timestamps`: DateTime stamps for each data point
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price  
- `close`: Closing price
- `volume`: Trading volume
- `amount`: Trading amount

(volume and amount can be 0 if not available)

### Sample Data Format

| timestamps | open | close | high | low | volume | amount |
|------------|------|-------|------|-----|--------|--------|
| 2019/11/26 9:35 | 182.45215 | 184.45215 | 184.95215 | 182.45215 | 15136000 | 0 |
| 2019/11/26 9:40 | 184.35215 | 183.85215 | 184.55215 | 183.45215 | 4433300 | 0 |
| 2019/11/26 9:45 | 183.85215 | 183.35215 | 183.95215 | 182.95215 | 3070900 | 0 |

> **Reference**: Check `data/HK_ali_09988_kline_5min_all.csv` for a complete example of the proper data format.


## 2. Config Preparation


Please edit the correct data path & pretrained model path and set your training parameters.

```yaml
# Data configuration
data:
  data_path: "/path/to/your/data.csv"
  lookback_window: 512        # Historical data points to use
  predict_window: 48           # Future points to predict
  max_context: 512            # Maximum context length

...

```
There are some other settings here, please see `configs/config_ali09988_candle-5min.yaml` for more comments.

## 3. Training

### Method 1: Sequential Training (Recommended)

The `train_sequential.py` script handles the complete training pipeline automatically:

```bash
# Complete training (tokenizer + predictor)
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml

# Skip existing models
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-existing

# Only train tokenizer
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-basemodel

# Only train predictor
python train_sequential.py --config configs/config_ali09988_candle-5min.yaml --skip-tokenizer
```

### Method 2: Individual Component Training

Train each component separately for more control:

```bash
# Step 1: Train tokenizer
python finetune_tokenizer.py --config configs/config_ali09988_candle-5min.yaml

# Step 2: Train predictor (requires fine-tuned tokenizer)
python finetune_base_model.py --config configs/config_ali09988_candle-5min.yaml
```

### DDP Training

For faster training on multiple GPUs:

```bash
# Set communication backend (nccl for NVIDIA GPUs, gloo for CPU/mixed)
DIST_BACKEND=nccl \
torchrun --standalone --nproc_per_node=8 train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

## 4. Training Results

The training process generates several outputs:

### Model Checkpoints
- **Tokenizer**: Saved to `{base_save_path}/{exp_name}/tokenizer/best_model/`
- **Predictor**: Saved to `{base_save_path}/{exp_name}/basemodel/best_model/`

### Training Logs
- **Console output**: Real-time training progress and metrics
- **Log files**: Detailed logs saved to `{base_save_path}/logs/`
- **Validation tracking**: Best models are saved based on validation loss

## 5. Prediction Vis

The following images show example training results on alibaba (HK stock) data:

![Training Result 1](examples/HK_ali_09988_kline_5min_all_historical_20250919_073929.png)

![Training Result 2](examples/HK_ali_09988_kline_5min_all_historical_20250919_073944.png)

![Training Result 3](examples/HK_ali_09988_kline_5min_all_historical_20250919_074012.png)

![Training Result 4](examples/HK_ali_09988_kline_5min_all_historical_20250919_074042.png)

![Training Result 5](examples/HK_ali_09988_kline_5min_all_historical_20250919_074251.png)

## 6. News + Kronos Fusion (Twitter / Fed / Multi-source)

This repo also includes a fusion pipeline that learns how historical news changes the next kline on top of Kronos outputs.

### 6.1 Prepare Historical News

Normalize raw files (Twitter/X exports, Fed releases, newswire CSVs, etc.) into one schema:

```bash
python3 finetune_csv/news_fusion/prepare_news_dataset.py \
  --inputs \
    path/to/twitter_history.csv \
    path/to/fed_releases.csv \
    path/to/newswire.csv \
  --sources twitter fed reuters \
  --output finetune_csv/data/news/news_normalized.csv
```

Output schema:
- `timestamp`
- `source`
- `title`
- `text`
- `url`
- `sentiment` (optional in raw input; auto-scored if missing)
- `impact` (optional, default 1.0)

Reference template: `data/news/news_template.csv`

### 6.2 Train Fusion Model

The fusion model uses:
- Kronos next-bar prediction features
- Rolling news features (counts/sentiment/source intensity)
- A residual MLP that corrects Kronos `open/high/low/close`

```bash
python3 finetune_csv/news_fusion/train_news_fusion.py train \
  --kline-path finetune_csv/data/BTCUSDT_kline_5min_2018_2025.csv \
  --news-path finetune_csv/data/news/news_normalized.csv \
  --model-id NeoQuasar/Kronos-base \
  --tokenizer-id NeoQuasar/Kronos-Tokenizer-base \
  --lookback 512 \
  --num-windows 12000 \
  --batch-size 32 \
  --device cuda:0 \
  --output-model finetune_csv/news_fusion/models/fusion_model.pt \
  --output-report finetune_csv/news_fusion/reports/fusion_train_report.json
```

### 6.3 Predict Next Kline (Kronos + Fused)

```bash
python3 finetune_csv/news_fusion/train_news_fusion.py predict-next \
  --kline-path finetune_csv/data/BTCUSDT_kline_5min_2026_to_now.csv \
  --news-path finetune_csv/data/news/news_normalized.csv \
  --fusion-model finetune_csv/news_fusion/models/fusion_model.pt \
  --device cuda:0 \
  --output-json finetune_csv/news_fusion/reports/next_kline_prediction.json
```

The output JSON includes:
- raw `kronos` prediction
- news-adjusted `fusion` prediction
- predicted residual adjustments and news diagnostics

## 7. Auto-Rotate Fine-Tuning Jobs

To keep fine-tuning running continuously, use the auto-cycle runner.
It launches jobs in order and, when one exits, starts the next automatically.

### Default rotation (1d -> 1h, forever)

```bash
python3 finetune_csv/auto_finetune_cycle.py
```

### Custom rotation

```bash
python3 finetune_csv/auto_finetune_cycle.py \
  --job "btc_1d::python3 train_sequential.py --config configs/config_btcusdt_1d_2018_2025.yaml" \
  --job "btc_1h::python3 train_sequential.py --config configs/config_btcusdt_1h_2018_2025.yaml" \
  --job "btc_5min::python3 train_sequential.py --config configs/config_btcusdt_5min_2018_2025.yaml"
```

Useful files:
- Runner PID: `finetune_csv/auto_finetune_cycle.pid`
- Current job metadata: `finetune_csv/auto_finetune_cycle.current`
- Runner log: `finetune_csv/logs/auto_finetune_cycle.log`
- Per-run logs: `finetune_csv/logs/auto_finetune_cycle/*.log`
- Run history: `finetune_csv/logs/auto_finetune_cycle_state.jsonl`

## 8. Run Once For Each Interval

If you want a single pass across the main BTCUSDT intervals instead of an endless rotation,
use `run_each_interval.py`.

### Default pass (5min -> 1h -> 1d)

```bash
python3 finetune_csv/run_each_interval.py
```

### Dry-run the commands first

```bash
python3 finetune_csv/run_each_interval.py --dry-run
```

### Run a subset and forward extra training flags

```bash
python3 finetune_csv/run_each_interval.py \
  --interval 5min \
  --interval 1h \
  --extra-arg=--skip-existing
```

### Override or add interval configs

```bash
python3 finetune_csv/run_each_interval.py \
  --interval 15m \
  --interval-config 15m=configs/config_btcusdt_15m_custom.yaml
```
