import os
import pandas as pd
import numpy as np
import json
import io
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
import time
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

try:
    from finetune_csv.download_klines import fetch_klines, to_kronos_rows, write_csv, INTERVAL_TO_MS, to_ms
    BINANCE_DOWNLOAD_AVAILABLE = True
except ImportError:
    BINANCE_DOWNLOAD_AVAILABLE = False
    print("Warning: Binance kline downloader cannot be imported, live market refresh will use local files only")

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None
loaded_model_key = None
loaded_model_info = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / 'reports'
FINETUNE_DIR = BASE_DIR / 'finetune_csv'
LIVE_POLL_SECONDS = 15

SITE_TIMEFRAMES = {
    '5min': {
        'slug': '5min',
        'display_interval': '5 min',
        'title': '5-minute native direction head',
        'short_title': '5min native head',
        'status_label': 'Best saved short-horizon model',
        'note_level': 'modest',
        'prediction_kind': 'binary_direction',
        'prediction_path': FINETUNE_DIR / 'runs_requested_by_user' / 'conference_report_20260319' / 'oldsplit_5min_2026_to_2026-03-09_predictions.csv',
        'actual_path': FINETUNE_DIR / 'data' / 'BTCUSDT_kline_5min_2026_to_now.csv',
        'live_actual_candidates': [
            FINETUNE_DIR / 'data' / 'BTCUSDT_kline_5min_last2m_live.csv',
            FINETUNE_DIR / 'data' / 'BTCUSDT_kline_5min_last3m_live_20260310_refresh.csv',
            FINETUNE_DIR / 'data' / 'BTCUSDT_kline_5min_2026_to_now.csv',
        ],
        'rolling_window': 288,
        'rolling_label': 'Rolling 24h accuracy',
        'detail_points': 720,
        'trade_style': 'Close-to-close direction',
        'caveat': 'Edge is positive but still narrow. This should be read as the best saved 5-minute result, not a runaway winner.',
    },
    '15min': {
        'slug': '15min',
        'display_interval': '15 min',
        'title': '15-minute day-focus adapter',
        'short_title': '15min adapter',
        'status_label': 'Best available 15min artifact',
        'note_level': 'limited',
        'prediction_kind': 'predicted_ohlc',
        'prediction_path': FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'eval_epoch1_2026_to_2026-03-21_results' / 'pred_15m.csv',
        'actual_path': FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'eval_epoch1_2026_to_2026-03-21_data' / 'BTCUSDT_kline_15m_2026_to_2026-03-21_exact.csv',
        'live_actual_candidates': [
            FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'eval_epoch1_2026_to_2026-03-21_data' / 'BTCUSDT_kline_15m_2026_to_2026-03-21_live.csv',
            FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'eval_epoch1_2026_to_2026-03-21_data' / 'BTCUSDT_kline_15m_2026_to_2026-03-21_exact.csv',
        ],
        'metrics_path': FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'eval_epoch1_2026_to_2026-03-21_results' / 'metrics_15m.json',
        'rolling_window': 96,
        'rolling_label': 'Rolling 24h accuracy',
        'detail_points': 384,
        'trade_style': 'Open-to-close candle direction',
        'caveat': 'The repo only contains one clean 15-minute evaluation snapshot. Accuracy stays slightly below the majority baseline, so this interval is still exploratory.',
    },
    '1day': {
        'slug': '1day',
        'display_interval': '1 day',
        'title': '1-day last6 path adapter',
        'short_title': '1day path adapter',
        'status_label': 'Best saved deployment result',
        'note_level': 'strong',
        'prediction_kind': 'path_close',
        'prediction_path': FINETUNE_DIR / 'runs_requested_by_user' / 'conference_report_20260319' / 'cons_only_lr3e6_last6_adapter_1d_close_to_close_fullrange_2026-03-16_rows.csv',
        'actual_path': FINETUNE_DIR / 'data' / 'BTCUSDT_kline_1d_2026_to_now.csv',
        'live_actual_candidates': [
            FINETUNE_DIR / 'data' / 'BTCUSDT_kline_1d_2026_to_now.csv',
            FINETUNE_DIR / 'data' / 'BTCUSDT_kline_1d_2017_to_2026_live.csv',
        ],
        'live_forecast_summary_path': FINETUNE_DIR / 'runs_requested_by_user' / 'pred_len_outputs_20260330' / 'summary.json',
        'rolling_window': 7,
        'rolling_label': 'Rolling 7-day accuracy',
        'detail_points': 120,
        'trade_style': 'Close-to-close path direction',
        'caveat': 'This is the strongest saved daily result in the repo, but later March refreshes show the edge compressing fast.',
    },
}

TIMEFRAME_ALIASES = {
    '5m': '5min',
    '5min': '5min',
    '15m': '15min',
    '15min': '15min',
    '1d': '1day',
    '1day': '1day',
    'day': '1day',
    'daily': '1day',
}

PROFIT_SUMMARY_ROW_NAMES = {
    '5min': 'direction_head_oldsplit_close_to_close_5min_refresh',
    '1day': 'cons_only_lr3e6_last6_adapter_1d_fullrange',
    '1h': 'direction_head_pruned50_actualfeed_1h',
}

DEFAULT_INTERVAL_DELTAS = {
    '5min': pd.Timedelta(minutes=5),
    '15min': pd.Timedelta(minutes=15),
    '1day': pd.Timedelta(days=1),
}

BASE_TOKENIZER_PATH = BASE_DIR / '.hf_cache' / 'hub' / 'models--NeoQuasar--Kronos-Tokenizer-base' / 'snapshots' / '0e0117387f39004a9016484a186a908917e22426'

LIVE_FINETUNED_SPECS = {
    '5min': {
        'model_key': 'btcusdt-5min-ft-live',
        'name': 'BTCUSDT 5min finetuned',
        'status_label': 'Finetuned live forecast',
        'model_path': FINETUNE_DIR / 'finetuned' / 'BTCUSDT_kline_5min_last6m' / 'basemodel' / 'best_model',
        'tokenizer_path': FINETUNE_DIR / 'finetuned' / 'BTCUSDT_kline_5min_last6m' / 'tokenizer' / 'best_model',
        'data_candidates': SITE_TIMEFRAMES['5min']['live_actual_candidates'],
        'max_context': 512,
        'lookback': 512,
        'pred_len': 8,
        'clip': 5,
        'top_k': 1,
        'top_p': 1.0,
        'sample_count': 1,
        'device': 'cpu',
        'direction_mode': 'close_to_close',
        'market_symbol': 'BTCUSDT',
        'binance_interval': '5m',
        'binance_fetch_rows': 576,
        'binance_refresh_seconds': 60,
        'binance_output_path': FINETUNE_DIR / 'data' / 'live_binance' / 'BTCUSDT_kline_5min_live_binance.csv',
        'params': '102.3M ft',
        'description': 'Local 5-minute finetuned checkpoint using the newest available BTCUSDT 5-minute candles in the repo.',
    },
    '15min': {
        'model_key': 'btcusdt-15min-ft-live',
        'name': 'BTCUSDT 15min finetuned',
        'status_label': 'Finetuned live forecast',
        'model_path': FINETUNE_DIR / 'runs_requested_by_user' / 'retrain_cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'finetuned' / 'cons_only_lr3e6_last2_adapter_dayfocus_nostab_20260321_201313' / 'basemodel' / 'best_model',
        'tokenizer_path': BASE_TOKENIZER_PATH,
        'data_candidates': SITE_TIMEFRAMES['15min']['live_actual_candidates'],
        'max_context': 256,
        'lookback': 256,
        'pred_len': 8,
        'clip': 5,
        'top_k': 1,
        'top_p': 1.0,
        'sample_count': 1,
        'device': 'cpu',
        'direction_mode': 'candle',
        'market_symbol': 'BTCUSDT',
        'binance_interval': '15m',
        'binance_fetch_rows': 320,
        'binance_refresh_seconds': 120,
        'binance_output_path': FINETUNE_DIR / 'data' / 'live_binance' / 'BTCUSDT_kline_15min_live_binance.csv',
        'params': '102.3M ft',
        'description': 'Local 15-minute day-focus adapter checkpoint applied to the newest saved 15-minute candle file.',
    },
    '1day': {
        'model_key': 'btcusdt-1day-ft-live',
        'name': 'BTCUSDT 1day finetuned',
        'status_label': 'Finetuned live forecast',
        'model_path': FINETUNE_DIR / 'runs_requested_by_user' / 'research_1m_multistep_from_epoch4_20260311' / 'screening_trials' / 'cons_only_lr3e6_last6_adapter' / 'finetuned' / 'cons_only_lr3e6_last6_adapter' / 'basemodel' / 'best_model',
        'tokenizer_path': BASE_TOKENIZER_PATH,
        'data_candidates': SITE_TIMEFRAMES['1day']['live_actual_candidates'],
        'max_context': 256,
        'lookback': 256,
        'pred_len': 5,
        'clip': 5,
        'top_k': 1,
        'top_p': 1.0,
        'sample_count': 1,
        'device': 'cpu',
        'direction_mode': 'close_to_close',
        'market_symbol': 'BTCUSDT',
        'binance_interval': '1d',
        'binance_fetch_rows': 300,
        'binance_refresh_seconds': 1800,
        'binance_output_path': FINETUNE_DIR / 'data' / 'live_binance' / 'BTCUSDT_kline_1day_live_binance.csv',
        'params': '102.3M ft',
        'description': 'Local daily finetuned path model using the freshest saved daily BTCUSDT candles in the repo.',
    },
}

for live_spec in LIVE_FINETUNED_SPECS.values():
    AVAILABLE_MODELS[live_spec['model_key']] = {
        'name': live_spec['name'],
        'model_id': str(live_spec['model_path']),
        'tokenizer_id': str(live_spec['tokenizer_path']),
        'context_length': live_spec['max_context'],
        'params': live_spec['params'],
        'description': live_spec['description'],
    }


def normalize_timeframe(value):
    """Normalize route and query aliases to internal timeframe slugs."""
    return TIMEFRAME_ALIASES.get(str(value).strip().lower(), '5min')


def serialize_float(value, digits=2):
    """Convert scalars to JSON-safe floats."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    value = float(value)
    return round(value, digits)


def serialize_int(value):
    """Convert scalars to JSON-safe ints."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


def serialize_timestamp(value):
    """Convert timestamps to ISO-8601 strings."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return pd.Timestamp(value).to_pydatetime().replace(tzinfo=None).isoformat()


def file_mtime_iso(path):
    """Return file modification time in ISO-8601 UTC."""
    if path is None or not Path(path).exists():
        return None
    return datetime.datetime.utcfromtimestamp(Path(path).stat().st_mtime).replace(microsecond=0).isoformat() + 'Z'


def current_utc_iso():
    """Return current UTC time in ISO-8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def resolve_artifact_path(path):
    """Resolve a checkpoint directory to the file that best represents its version."""
    if path is None:
        return None
    artifact_path = Path(path)
    if artifact_path.is_dir():
        for filename in ['model.safetensors', 'pytorch_model.bin', 'config.json']:
            candidate = artifact_path / filename
            if candidate.exists():
                return candidate
    return artifact_path


def artifact_mtime_iso(path):
    """Return the version timestamp for a file or checkpoint directory."""
    return file_mtime_iso(resolve_artifact_path(path))


def artifact_signature(path):
    """Return a cache key that changes when an artifact changes."""
    artifact_path = resolve_artifact_path(path)
    if artifact_path is None or not artifact_path.exists():
        return 'missing'
    stat = artifact_path.stat()
    return f'{artifact_path}:{stat.st_mtime_ns}:{stat.st_size}'


def combine_error_messages(*messages):
    """Combine multiple optional error messages into one string."""
    cleaned = [str(message).strip() for message in messages if message]
    return ' | '.join(cleaned) if cleaned else None


def market_source_label(source_code):
    """Return a human-readable market source label."""
    labels = {
        'binance_api': 'Binance API',
        'binance_cache': 'Cached Binance file',
        'local_fallback': 'Local fallback file',
        'local_file': 'Local repo file',
    }
    return labels.get(source_code, 'Unknown source')


def choose_latest_existing(candidates):
    """Return the newest existing path from a candidate list."""
    existing = [Path(candidate) for candidate in candidates if Path(candidate).exists()]
    if not existing:
        return None
    return max(existing, key=lambda candidate: candidate.stat().st_mtime)


@lru_cache(maxsize=None)
def load_csv_cached(path_str):
    """Load a CSV file once and reuse it across requests."""
    return pd.read_csv(path_str)


def load_csv(path):
    """Load a CSV file and return a defensive copy."""
    return load_csv_cached(str(path)).copy()


@lru_cache(maxsize=1)
def load_selected_winners_summary():
    """Load the curated cross-timeframe winner table."""
    return pd.read_csv(REPORTS_DIR / 'sgp2_extension_20260405' / 'selected_winners_summary.csv')


@lru_cache(maxsize=1)
def load_profit_summary():
    """Load the deployment profit summary JSON."""
    with open(FINETUNE_DIR / 'runs_requested_by_user' / 'conference_report_20260319' / 'final_candidate_profit_summary.json', 'r', encoding='utf-8') as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_daily_screening_summary():
    """Load the daily ablation table."""
    return pd.read_csv(REPORTS_DIR / 'sgp2_extension_20260405' / 'daily_screening_summary.csv')


@lru_cache(maxsize=1)
def load_daily_winner_extension():
    """Load the daily extension table."""
    return pd.read_csv(REPORTS_DIR / 'sgp2_extension_20260405' / 'daily_winner_extension.csv')


@lru_cache(maxsize=1)
def load_hourly_pruning_summary():
    """Load the hourly pruning sweep table."""
    return pd.read_csv(REPORTS_DIR / 'sgp2_extension_20260405' / 'hourly_pruning_sweep.csv')


@lru_cache(maxsize=1)
def load_legacy_5min_baselines():
    """Load the legacy 5-minute baseline table."""
    return pd.read_csv(REPORTS_DIR / 'sgp2_extension_20260405' / 'legacy_5min_baselines.csv')


@lru_cache(maxsize=1)
def load_15min_metrics():
    """Load the saved 15-minute metrics JSON."""
    with open(SITE_TIMEFRAMES['15min']['metrics_path'], 'r', encoding='utf-8') as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_daily_live_forecast_summary():
    """Load the saved daily forward forecast JSON."""
    with open(SITE_TIMEFRAMES['1day']['live_forecast_summary_path'], 'r', encoding='utf-8') as handle:
        return json.load(handle)


def standardize_ohlc_dataframe(df):
    """Normalize OHLC dataframes to a shared timestamp column."""
    df = df.copy()
    timestamp_column = None
    for candidate in ['timestamps', 'timestamp', 'target_timestamp', 'timestamp_str', 'date']:
        if candidate in df.columns:
            timestamp_column = candidate
            break
    if timestamp_column is None:
        raise ValueError('No timestamp column found in OHLC dataframe')
    df['timestamp'] = pd.to_datetime(df[timestamp_column], errors='coerce')
    for column in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
    return df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)


@lru_cache(maxsize=None)
def load_ohlc_dataframe(path_str):
    """Load and normalize an OHLC CSV."""
    return standardize_ohlc_dataframe(pd.read_csv(path_str))


def get_ohlc_dataframe(path):
    """Return a defensive copy of a normalized OHLC dataframe."""
    return load_ohlc_dataframe(str(path)).copy()


def infer_time_delta(timestamps, fallback=None):
    """Infer the candle spacing from the most recent timestamps."""
    timestamp_series = pd.Series(pd.to_datetime(timestamps, errors='coerce')).dropna().sort_values().reset_index(drop=True)
    diffs = timestamp_series.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if not diffs.empty:
        try:
            return diffs.mode().iloc[0]
        except (IndexError, ValueError):
            return diffs.iloc[-1]
    return fallback


def build_prediction_input_frame(df):
    """Select the predictor input columns from a normalized market dataframe."""
    columns = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        columns.append('volume')
    if 'amount' in df.columns:
        columns.append('amount')
    return df[columns].copy()


def sanitize_forecast_frame(df):
    """Normalize and validate a predicted OHLC frame."""
    forecast_df = df.copy()
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'], errors='coerce')
    for column in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if column in forecast_df.columns:
            forecast_df[column] = pd.to_numeric(forecast_df[column], errors='coerce')
    forecast_df = forecast_df.dropna(subset=['timestamp', 'open', 'high', 'low', 'close']).sort_values('timestamp').reset_index(drop=True)
    forecast_df['high'] = forecast_df[['open', 'high', 'low', 'close']].max(axis=1)
    forecast_df['low'] = forecast_df[['open', 'high', 'low', 'close']].min(axis=1)
    if (forecast_df[['open', 'high', 'low', 'close']] <= 0).any().any():
        raise ValueError('Finetuned forecast produced non-positive prices')
    return forecast_df


def apply_live_direction_rule(slug, forecast_df, last_close):
    """Attach live direction labels and move percentages to a forecast frame."""
    forecast_df = forecast_df.copy()
    direction_mode = LIVE_FINETUNED_SPECS[slug]['direction_mode']

    if direction_mode == 'close_to_close':
        reference_close = float(last_close)
        reference_values = []
        move_pcts = []
        pred_dirs = []
        for _, row in forecast_df.iterrows():
            reference_values.append(reference_close)
            pred_close = float(row['close'])
            pred_dirs.append(1 if pred_close >= reference_close else 0)
            move_pcts.append(((pred_close / reference_close) - 1.0) * 100 if reference_close else None)
            reference_close = pred_close
        forecast_df['reference_close'] = reference_values
        forecast_df['pred_dir'] = pred_dirs
        forecast_df['move_pct'] = move_pcts
        return forecast_df

    base_open = forecast_df['open'].replace(0, np.nan)
    forecast_df['pred_dir'] = (forecast_df['close'] >= forecast_df['open']).astype(int)
    forecast_df['move_pct'] = ((forecast_df['close'] / base_open) - 1.0) * 100
    return forecast_df


@lru_cache(maxsize=16)
def refresh_binance_market_file_cached(slug, refresh_bucket):
    """Download recent Binance klines for a timeframe and save them in Kronos CSV format."""
    del refresh_bucket
    if not BINANCE_DOWNLOAD_AVAILABLE:
        raise RuntimeError('Binance downloader not available in this environment')

    live_spec = LIVE_FINETUNED_SPECS[slug]
    interval = live_spec['binance_interval']
    interval_ms = INTERVAL_TO_MS[interval]
    fetch_rows = max(live_spec['binance_fetch_rows'], live_spec['lookback'] + live_spec['pred_len'] + 2)
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    end_ms = to_ms(end_dt)
    start_ms = end_ms - (fetch_rows * interval_ms)

    with io.StringIO() as _binance_stdout, redirect_stdout(_binance_stdout):
        raw_rows = fetch_klines(
            symbol=live_spec['market_symbol'],
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    if raw_rows and int(raw_rows[-1][0]) + interval_ms > end_ms:
        raw_rows = raw_rows[:-1]
    if len(raw_rows) < live_spec['lookback']:
        raise ValueError(f'Binance returned only {len(raw_rows)} rows for {slug}, need at least {live_spec["lookback"]}')

    output_path = Path(live_spec['binance_output_path'])
    write_csv(output_path, to_kronos_rows(raw_rows), include_direction=False)
    return output_path


def refresh_binance_market_file(slug, force=False):
    """Refresh Binance market data for a timeframe using a short per-interval cache window."""
    refresh_seconds = LIVE_FINETUNED_SPECS[slug]['binance_refresh_seconds']
    refresh_bucket = time.time_ns() if force else int(time.time() // refresh_seconds)
    return refresh_binance_market_file_cached(slug, refresh_bucket)


def resolve_live_market_context(slug, force_refresh=False):
    """Resolve the market file to use for live forecasting, preferring Binance refreshes."""
    live_spec = LIVE_FINETUNED_SPECS[slug]
    local_path = choose_latest_existing(live_spec['data_candidates']) or choose_latest_existing(SITE_TIMEFRAMES[slug]['live_actual_candidates']) or SITE_TIMEFRAMES[slug]['actual_path']
    binance_output_path = Path(live_spec['binance_output_path'])

    if live_spec.get('binance_interval'):
        try:
            refreshed_path = refresh_binance_market_file(slug, force=force_refresh)
            return {
                'path': refreshed_path,
                'source': 'binance_api',
                'source_label': market_source_label('binance_api'),
                'error': None,
            }
        except Exception as exc:
            if binance_output_path.exists():
                return {
                    'path': binance_output_path,
                    'source': 'binance_cache',
                    'source_label': market_source_label('binance_cache'),
                    'error': f'Binance refresh failed, using cached Binance file: {exc}',
                }
            if local_path:
                return {
                    'path': local_path,
                    'source': 'local_fallback',
                    'source_label': market_source_label('local_fallback'),
                    'error': f'Binance refresh failed, using local fallback file: {exc}',
                }
            raise

    if local_path:
        return {
            'path': local_path,
            'source': 'local_file',
            'source_label': market_source_label('local_file'),
            'error': None,
        }
    raise FileNotFoundError(f'No live market file found for {slug}')


@lru_cache(maxsize=8)
def load_finetuned_predictor_cached(model_path_str, tokenizer_path_str, max_context, clip, device, model_signature, tokenizer_signature):
    """Load and cache a finetuned predictor for live inference."""
    del model_signature, tokenizer_signature
    if not MODEL_AVAILABLE:
        raise RuntimeError('Kronos model library not available')
    loaded_tokenizer = KronosTokenizer.from_pretrained(tokenizer_path_str)
    loaded_model = Kronos.from_pretrained(model_path_str)
    loaded_tokenizer.eval()
    loaded_model.eval()
    return KronosPredictor(loaded_model, loaded_tokenizer, device=device, max_context=max_context, clip=clip)


def get_live_finetuned_predictor(slug):
    """Return the cached finetuned predictor for a timeframe."""
    live_spec = LIVE_FINETUNED_SPECS[slug]
    return load_finetuned_predictor_cached(
        str(live_spec['model_path']),
        str(live_spec['tokenizer_path']),
        live_spec['max_context'],
        live_spec['clip'],
        live_spec['device'],
        artifact_signature(live_spec['model_path']),
        artifact_signature(live_spec['tokenizer_path']),
    )


@lru_cache(maxsize=16)
def build_live_finetuned_forecast_cached(
    slug,
    data_path_str,
    data_signature,
    model_signature,
    tokenizer_signature,
    lookback,
    pred_len,
    top_k,
    top_p,
    sample_count,
):
    """Run finetuned live inference for a timeframe and cache it until inputs change."""
    del data_signature
    live_spec = LIVE_FINETUNED_SPECS[slug]
    market_df = get_ohlc_dataframe(data_path_str)
    if len(market_df) < 2:
        raise ValueError(f'Not enough rows in live data file for {slug}')

    context_df = market_df.tail(min(lookback, len(market_df))).copy()
    time_delta = infer_time_delta(context_df['timestamp'], DEFAULT_INTERVAL_DELTAS[slug])
    if time_delta is None:
        raise ValueError(f'Unable to infer candle spacing for {slug}')

    y_timestamp = pd.Series(pd.date_range(
        start=context_df['timestamp'].iloc[-1] + time_delta,
        periods=pred_len,
        freq=time_delta,
    ))

    predictor_instance = load_finetuned_predictor_cached(
        str(live_spec['model_path']),
        str(live_spec['tokenizer_path']),
        live_spec['max_context'],
        live_spec['clip'],
        live_spec['device'],
        model_signature,
        tokenizer_signature,
    )
    predicted_df = predictor_instance.predict(
        build_prediction_input_frame(context_df),
        context_df['timestamp'],
        y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_k=top_k,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )
    forecast_df = sanitize_forecast_frame(
        predicted_df.reset_index().rename(columns={'index': 'timestamp'})
    )
    forecast_df = apply_live_direction_rule(slug, forecast_df, context_df['close'].iloc[-1])

    next_row = forecast_df.iloc[0]
    generated_at = current_utc_iso()
    return {
        'model_name': live_spec['name'],
        'model_path': str(live_spec['model_path']),
        'model_path_mtime': artifact_mtime_iso(live_spec['model_path']),
        'data_path': data_path_str,
        'data_path_mtime': file_mtime_iso(data_path_str),
        'generated_at': generated_at,
        'context_end_utc': serialize_timestamp(context_df['timestamp'].iloc[-1]),
        'lookback_used': len(context_df),
        'pred_len': pred_len,
        'next_signal': {
            'timestamp': serialize_timestamp(next_row['timestamp']),
            'direction': 'Up' if int(next_row['pred_dir']) == 1 else 'Down',
            'confidence_pct': None,
            'move_pct': serialize_float(next_row.get('move_pct')),
            'predicted_close': serialize_float(next_row.get('close')),
            'predicted_open': serialize_float(next_row.get('open')),
            'predicted_high': serialize_float(next_row.get('high')),
            'predicted_low': serialize_float(next_row.get('low')),
            'actual_close': serialize_float(context_df['close'].iloc[-1]),
        },
        'forecast_rows': [
            {
                'timestamp': serialize_timestamp(row['timestamp']),
                'direction': 'Up' if int(row['pred_dir']) == 1 else 'Down',
                'open': serialize_float(row['open']),
                'high': serialize_float(row['high']),
                'low': serialize_float(row['low']),
                'close': serialize_float(row['close']),
                'move_pct': serialize_float(row.get('move_pct')),
            }
            for _, row in forecast_df.head(8).iterrows()
        ],
    }


def build_live_finetuned_forecast(slug, data_path=None):
    """Build a live finetuned forecast payload for a timeframe."""
    live_spec = LIVE_FINETUNED_SPECS[slug]
    data_path = Path(data_path) if data_path is not None else resolve_live_market_context(slug)['path']
    if data_path is None or not Path(data_path).exists():
        raise FileNotFoundError(f'No live data file found for {slug}')
    return build_live_finetuned_forecast_cached(
        slug,
        str(data_path),
        artifact_signature(data_path),
        artifact_signature(live_spec['model_path']),
        artifact_signature(live_spec['tokenizer_path']),
        live_spec['lookback'],
        live_spec['pred_len'],
        live_spec['top_k'],
        live_spec['top_p'],
        live_spec['sample_count'],
    )


def get_profit_summary_row(name):
    """Fetch a specific row from the deployment profit summary."""
    for row in load_profit_summary():
        if row.get('name') == name:
            return row
    return None


def get_selected_winner_row(interval_label):
    """Fetch a specific row from the curated winner summary."""
    summary = load_selected_winners_summary()
    matches = summary[summary['interval_label'] == interval_label]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_hourly_reference_summary():
    """Return the strong published hourly benchmark as context for the 15-minute caveat."""
    winner_row = get_selected_winner_row('1h') or {}
    profit_row = get_profit_summary_row(PROFIT_SUMMARY_ROW_NAMES['1h']) or {}
    return {
        'title': winner_row.get('model_label', '1-hour pruned50 direction head'),
        'accuracy_pct': serialize_float(winner_row.get('accuracy_pct')),
        'baseline_pct': serialize_float(winner_row.get('baseline_pct')),
        'delta_vs_baseline_pp': serialize_float(winner_row.get('delta_vs_baseline_pp')),
        'total_pnl_usdt': serialize_float(winner_row.get('total_pnl_usdt')),
        'compounded_return_pct': serialize_float(winner_row.get('compounded_return_pct')),
        'predicted_up_ratio_pct': serialize_float((profit_row.get('pred_up_ratio') or 0) * 100),
        'actual_up_ratio_pct': serialize_float((profit_row.get('actual_up_ratio') or 0) * 100),
    }


def build_timeframe_summary(slug):
    """Build the summary card payload for a timeframe."""
    spec = SITE_TIMEFRAMES[slug]
    if slug == '15min':
        metrics = load_15min_metrics()
        accuracy_pct = metrics['directional_accuracy_candle_close_vs_open'] * 100
        baseline_pct = metrics['majority_baseline_accuracy'] * 100
        predicted_up_ratio_pct = metrics['predicted_up_ratio'] * 100
        actual_up_ratio_pct = metrics['actual_up_ratio'] * 100
        pnl_proxy = None
        compounded_proxy = None
        rationale = 'Only clean saved 15-minute artifact in the repo. Useful as a live middle-horizon monitor, but still too close to baseline to claim a robust edge.'
        return {
            'slug': slug,
            'display_interval': spec['display_interval'],
            'title': spec['title'],
            'short_title': spec['short_title'],
            'status_label': spec['status_label'],
            'note_level': spec['note_level'],
            'accuracy_pct': serialize_float(accuracy_pct),
            'baseline_pct': serialize_float(baseline_pct),
            'delta_vs_baseline_pp': serialize_float(accuracy_pct - baseline_pct),
            'total_pnl_usdt': pnl_proxy,
            'compounded_return_pct': compounded_proxy,
            'predicted_up_ratio_pct': serialize_float(predicted_up_ratio_pct),
            'actual_up_ratio_pct': serialize_float(actual_up_ratio_pct),
            'bias_gap_pp': serialize_float(predicted_up_ratio_pct - actual_up_ratio_pct),
            'rows': serialize_int(metrics.get('rows')),
            'range_start': metrics.get('range_pred_start'),
            'range_end': metrics.get('range_pred_end'),
            'trade_style': spec['trade_style'],
            'rationale': rationale,
            'caveat': spec['caveat'],
            'source_path': str(spec['prediction_path']),
        }

    interval_label = '5min' if slug == '5min' else '1d'
    winner_row = get_selected_winner_row(interval_label) or {}
    profit_row = get_profit_summary_row(PROFIT_SUMMARY_ROW_NAMES[slug]) or {}
    actual_up_ratio_pct = serialize_float((profit_row.get('actual_up_ratio') or 0) * 100)
    predicted_up_ratio_pct = serialize_float((profit_row.get('pred_up_ratio') or 0) * 100)
    baseline_pct = serialize_float(max(
        (profit_row.get('actual_up_ratio') or 0) * 100,
        100 - ((profit_row.get('actual_up_ratio') or 0) * 100),
    ))
    accuracy_pct = serialize_float((profit_row.get('accuracy') or 0) * 100)
    return {
        'slug': slug,
        'display_interval': spec['display_interval'],
        'title': spec['title'],
        'short_title': spec['short_title'],
        'status_label': spec['status_label'],
        'note_level': spec['note_level'],
        'accuracy_pct': accuracy_pct,
        'baseline_pct': baseline_pct,
        'delta_vs_baseline_pp': serialize_float(accuracy_pct - baseline_pct),
        'total_pnl_usdt': serialize_float(profit_row.get('total_pnl_1btc_usdt')),
        'compounded_return_pct': serialize_float(profit_row.get('compounded_return_pct_from_1.0')),
        'predicted_up_ratio_pct': predicted_up_ratio_pct,
        'actual_up_ratio_pct': actual_up_ratio_pct,
        'bias_gap_pp': serialize_float(predicted_up_ratio_pct - actual_up_ratio_pct),
        'rows': serialize_int(profit_row.get('rows') or winner_row.get('rows')),
        'range_start': profit_row.get('range_start') or winner_row.get('range_start'),
        'range_end': profit_row.get('range_end') or winner_row.get('range_end'),
        'trade_style': spec['trade_style'],
        'rationale': winner_row.get('rationale', ''),
        'caveat': spec['caveat'],
        'source_path': str(spec['prediction_path']),
    }


def build_site_summary():
    """Return the summary cards for all website timeframes."""
    return [build_timeframe_summary(slug) for slug in ['5min', '15min', '1day']]


@lru_cache(maxsize=None)
def load_model_frame_cached(slug):
    """Load a standardized evaluation dataframe for a timeframe."""
    spec = SITE_TIMEFRAMES[slug]

    if spec['prediction_kind'] == 'binary_direction':
        prediction_df = load_csv(spec['prediction_path'])
        if slug == '5min':
            prediction_df['timestamp'] = pd.to_datetime(prediction_df['timestamp'], errors='coerce')
            prediction_df['actual_dir'] = pd.to_numeric(prediction_df['label'], errors='coerce')
            prediction_df['pred_dir'] = pd.to_numeric(prediction_df['pred_dir'], errors='coerce')
            prediction_df['pred_prob_up'] = pd.to_numeric(prediction_df['pred_prob_up'], errors='coerce')
            prediction_df['scored'] = pd.to_numeric(prediction_df.get('mask', 1), errors='coerce').fillna(1).astype(int)
            prediction_df['actual_return_pct'] = pd.to_numeric(prediction_df.get('ret_horizon'), errors='coerce') * 100
        else:
            prediction_df['timestamp'] = pd.to_datetime(prediction_df['target_timestamp'], errors='coerce')
            prediction_df['actual_dir'] = pd.to_numeric(prediction_df['actual_open_vs_close_dir'], errors='coerce')
            prediction_df['pred_dir'] = pd.to_numeric(prediction_df['pred_dir'], errors='coerce')
            prediction_df['pred_prob_up'] = pd.to_numeric(prediction_df['pred_prob_up'], errors='coerce')
            prediction_df['scored'] = pd.to_numeric(prediction_df.get('mask_dataset', 1), errors='coerce').fillna(1).astype(int)

        actual_df = get_ohlc_dataframe(spec['actual_path'])[['timestamp', 'open', 'high', 'low', 'close']]
        merged = prediction_df.merge(actual_df, on='timestamp', how='left')
        merged['actual_close'] = merged['close']
        merged['predicted_close'] = np.nan

    elif spec['prediction_kind'] == 'predicted_ohlc':
        pred_df = get_ohlc_dataframe(spec['prediction_path'])
        actual_df = get_ohlc_dataframe(spec['actual_path'])
        merged = actual_df[['timestamp', 'open', 'high', 'low', 'close']].merge(
            pred_df[['timestamp', 'open', 'high', 'low', 'close']],
            on='timestamp',
            how='inner',
            suffixes=('_actual', '_pred'),
        )
        merged['actual_dir'] = (merged['close_actual'] >= merged['open_actual']).astype(int)
        merged['pred_dir'] = (merged['close_pred'] >= merged['open_pred']).astype(int)
        merged['pred_prob_up'] = np.nan
        merged['scored'] = 1
        merged['open'] = merged['open_actual']
        merged['high'] = merged['high_actual']
        merged['low'] = merged['low_actual']
        merged['close'] = merged['close_actual']
        merged['actual_close'] = merged['close_actual']
        merged['predicted_close'] = merged['close_pred']

    else:
        prediction_df = load_csv(spec['prediction_path'])
        prediction_df['timestamp'] = pd.to_datetime(prediction_df['timestamp_str'], errors='coerce')
        prediction_df['pred_dir'] = pd.to_numeric(prediction_df['pred_close_to_close_direction'], errors='coerce')
        prediction_df['actual_dir'] = pd.to_numeric(prediction_df['actual_close_to_close_direction'], errors='coerce')
        prediction_df['actual_close'] = pd.to_numeric(prediction_df['actual_close'], errors='coerce')
        prediction_df['predicted_close'] = pd.to_numeric(prediction_df['pred_close'], errors='coerce')
        prediction_df['pred_prob_up'] = np.nan
        prediction_df['scored'] = 1
        prediction_df['pnl_usdt'] = pd.to_numeric(prediction_df['close_to_close_pnl_1btc_usdt'], errors='coerce')
        actual_df = get_ohlc_dataframe(spec['actual_path'])[['timestamp', 'open', 'high', 'low', 'close']]
        merged = prediction_df.merge(actual_df, on='timestamp', how='left', suffixes=('', '_ohlc'))
        merged['actual_close'] = merged['actual_close'].fillna(merged['close'])

    merged = merged.dropna(subset=['timestamp', 'pred_dir', 'actual_dir']).sort_values('timestamp').reset_index(drop=True)
    merged['correct'] = (merged['pred_dir'].astype(int) == merged['actual_dir'].astype(int)).astype(int)
    return merged


def get_model_frame(slug):
    """Return a defensive copy of a standardized model dataframe."""
    return load_model_frame_cached(slug).copy()


def build_confusion_matrix(df):
    """Build a confusion matrix from standardized model data."""
    tp = int(((df['pred_dir'] == 1) & (df['actual_dir'] == 1)).sum())
    tn = int(((df['pred_dir'] == 0) & (df['actual_dir'] == 0)).sum())
    fp = int(((df['pred_dir'] == 1) & (df['actual_dir'] == 0)).sum())
    fn = int(((df['pred_dir'] == 0) & (df['actual_dir'] == 1)).sum())
    total = tp + tn + fp + fn
    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'total': total,
        'precision_pct': serialize_float((tp / (tp + fp)) * 100 if (tp + fp) else None),
        'recall_pct': serialize_float((tp / (tp + fn)) * 100 if (tp + fn) else None),
    }


def build_monthly_breakdown(df):
    """Summarize accuracy by month for the detail page."""
    monthly = df.copy()
    monthly['month'] = monthly['timestamp'].dt.strftime('%Y-%m')
    breakdown = (
        monthly.groupby('month')
        .agg(
            rows=('correct', 'size'),
            accuracy_pct=('correct', lambda values: values.mean() * 100),
            predicted_up_ratio_pct=('pred_dir', lambda values: values.mean() * 100),
            actual_up_ratio_pct=('actual_dir', lambda values: values.mean() * 100),
        )
        .reset_index()
    )
    rows = []
    for _, row in breakdown.iterrows():
        rows.append({
            'month': row['month'],
            'rows': serialize_int(row['rows']),
            'accuracy_pct': serialize_float(row['accuracy_pct']),
            'predicted_up_ratio_pct': serialize_float(row['predicted_up_ratio_pct']),
            'actual_up_ratio_pct': serialize_float(row['actual_up_ratio_pct']),
        })
    return rows


def build_recent_rows(df, limit=12):
    """Build the recent prediction tape for the detail page."""
    recent = df.tail(limit).iloc[::-1]
    rows = []
    for _, row in recent.iterrows():
        confidence_pct = None
        if 'pred_prob_up' in recent.columns and not pd.isna(row.get('pred_prob_up')):
            confidence_pct = float(row['pred_prob_up']) * 100 if row['pred_dir'] == 1 else (1 - float(row['pred_prob_up'])) * 100
        rows.append({
            'timestamp': serialize_timestamp(row['timestamp']),
            'predicted_direction': 'Up' if int(row['pred_dir']) == 1 else 'Down',
            'actual_direction': 'Up' if int(row['actual_dir']) == 1 else 'Down',
            'correct': bool(row['correct']),
            'confidence_pct': serialize_float(confidence_pct),
            'actual_close': serialize_float(row.get('actual_close')),
            'predicted_close': serialize_float(row.get('predicted_close')),
            'pnl_usdt': serialize_float(row.get('pnl_usdt')),
        })
    return rows


def build_detail_payload(slug):
    """Build the chart and analysis payload for the detail page."""
    spec = SITE_TIMEFRAMES[slug]
    summary = build_timeframe_summary(slug)
    df = get_model_frame(slug)
    window_size = spec['rolling_window']
    detail_points = min(spec['detail_points'], len(df))
    detail_df = df.tail(detail_points).copy()
    detail_df['rolling_accuracy_pct'] = detail_df['correct'].rolling(window_size, min_periods=max(3, window_size // 4)).mean() * 100
    detail_df['marker_y'] = detail_df['actual_close'].fillna(detail_df['close']) * np.where(detail_df['pred_dir'] == 1, 1.004, 0.996)

    summary['observed_accuracy_pct'] = serialize_float(df['correct'].mean() * 100)
    summary['predicted_up_ratio_pct'] = serialize_float(df['pred_dir'].mean() * 100)
    summary['actual_up_ratio_pct'] = serialize_float(df['actual_dir'].mean() * 100)
    summary['bias_gap_pp'] = serialize_float(summary['predicted_up_ratio_pct'] - summary['actual_up_ratio_pct'])

    return {
        'summary': summary,
        'chart': {
            'timestamps': [serialize_timestamp(value) for value in detail_df['timestamp']],
            'actual_close': [serialize_float(value, 4) for value in detail_df['actual_close'].fillna(detail_df['close'])],
            'predicted_close': [serialize_float(value, 4) for value in detail_df['predicted_close']] if 'predicted_close' in detail_df.columns else [],
            'marker_y': [serialize_float(value, 4) for value in detail_df['marker_y']],
            'pred_dir': [serialize_int(value) for value in detail_df['pred_dir']],
            'actual_dir': [serialize_int(value) for value in detail_df['actual_dir']],
            'correct': [bool(value) for value in detail_df['correct']],
            'pred_prob_up': [serialize_float(value, 4) for value in detail_df['pred_prob_up']] if 'pred_prob_up' in detail_df.columns else [],
        },
        'rolling_chart': {
            'timestamps': [serialize_timestamp(value) for value in detail_df['timestamp']],
            'accuracy_pct': [serialize_float(value) for value in detail_df['rolling_accuracy_pct']],
            'baseline_pct': summary['baseline_pct'],
            'label': spec['rolling_label'],
        },
        'confusion': build_confusion_matrix(df),
        'monthly_breakdown': build_monthly_breakdown(df),
        'recent_rows': build_recent_rows(df),
    }


def build_table_rows(dataframe, limit=None):
    """Convert a dataframe to JSON-safe records."""
    if limit is not None:
        dataframe = dataframe.head(limit)
    records = dataframe.replace({np.nan: None}).to_dict(orient='records')
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, (pd.Timestamp, datetime.datetime)):
                record[key] = serialize_timestamp(value)
            elif isinstance(value, (np.floating, float)):
                record[key] = serialize_float(value)
            elif isinstance(value, (np.integer, int)):
                record[key] = serialize_int(value)
    return records


def build_overview_context():
    """Build the server-rendered context for the overview page."""
    models = build_site_summary()
    return {
        'models': models,
        'chart_rows': [
            {
                'label': model['display_interval'],
                'accuracy_pct': model['accuracy_pct'],
                'baseline_pct': model['baseline_pct'],
                'pnl_usdt': model['total_pnl_usdt'],
            }
            for model in models
        ],
        'hourly_reference': get_hourly_reference_summary(),
        'hourly_pruning_rows': build_table_rows(load_hourly_pruning_summary()),
        'daily_screening_rows': build_table_rows(load_daily_screening_summary()),
        'daily_extension_rows': build_table_rows(load_daily_winner_extension()),
        'legacy_5min_rows': build_table_rows(load_legacy_5min_baselines()),
        'generated_at': datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
    }


def minutes_since(timestamp_value):
    """Return elapsed minutes between now and a timestamp."""
    if timestamp_value is None:
        return None
    ts = pd.Timestamp(timestamp_value).to_pydatetime().replace(tzinfo=None)
    delta = datetime.datetime.utcnow().replace(microsecond=0) - ts
    return round(delta.total_seconds() / 60, 1)


def build_live_market_payload(path, points=48):
    """Load the most recent market tape for a live card."""
    if path is None:
        return None
    market_df = get_ohlc_dataframe(path).tail(points)
    if market_df.empty:
        return None
    last_row = market_df.iloc[-1]
    return {
        'path': str(path),
        'path_mtime': file_mtime_iso(path),
        'timestamps': [serialize_timestamp(value) for value in market_df['timestamp']],
        'closes': [serialize_float(value, 4) for value in market_df['close']],
        'last': {
            'timestamp': serialize_timestamp(last_row['timestamp']),
            'open': serialize_float(last_row['open']),
            'high': serialize_float(last_row['high']),
            'low': serialize_float(last_row['low']),
            'close': serialize_float(last_row['close']),
        },
    }


def build_daily_live_forecast():
    """Build the live daily forward tape."""
    summary = load_daily_live_forecast_summary()
    runs = summary.get('runs', {})
    if not runs:
        return None
    selected_key = max(runs.keys(), key=lambda value: int(value))
    selected_run = runs[selected_key]
    rows = selected_run.get('rows', [])
    if not rows:
        return None
    forecast_df = pd.DataFrame(rows)
    forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'], errors='coerce')
    for column in ['open', 'high', 'low', 'close']:
        forecast_df[column] = pd.to_numeric(forecast_df[column], errors='coerce')
    forecast_df = forecast_df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    forecast_df['pred_dir'] = (forecast_df['close'] >= forecast_df['open']).astype(int)
    next_row = forecast_df.iloc[0]
    return {
        'path': selected_run.get('csv_path'),
        'path_mtime': file_mtime_iso(selected_run.get('csv_path')),
        'generated_at': summary.get('generated_at_utc'),
        'context_end_utc': summary.get('context_end_utc'),
        'next_signal': {
            'timestamp': serialize_timestamp(next_row['timestamp']),
            'direction': 'Up' if int(next_row['pred_dir']) == 1 else 'Down',
            'open': serialize_float(next_row['open']),
            'high': serialize_float(next_row['high']),
            'low': serialize_float(next_row['low']),
            'close': serialize_float(next_row['close']),
        },
        'forecast_rows': [
            {
                'timestamp': serialize_timestamp(row['timestamp']),
                'direction': 'Up' if int(row['pred_dir']) == 1 else 'Down',
                'open': serialize_float(row['open']),
                'close': serialize_float(row['close']),
            }
            for _, row in forecast_df.head(8).iterrows()
        ],
    }


def build_live_snapshot():
    """Build the live polling payload for all three cards."""
    load_csv_cached.cache_clear()
    load_ohlc_dataframe.cache_clear()
    load_model_frame_cached.cache_clear()
    load_daily_live_forecast_summary.cache_clear()

    cards = []
    for slug in ['5min', '15min', '1day']:
        spec = SITE_TIMEFRAMES[slug]
        market_context = None
        forecast_error = None
        prediction_path = None
        prediction_timestamp = None
        forecast_rows = []
        source_generated_at = None
        signal = {
            'timestamp': None,
            'direction': 'Unavailable',
            'confidence_pct': None,
            'move_pct': None,
            'predicted_close': None,
            'actual_close': None,
        }
        status_label = spec['status_label']
        data_path = choose_latest_existing(spec['live_actual_candidates']) or spec['actual_path']
        market_source = 'local_file'
        market_source_label_value = market_source_label(market_source)

        if slug in LIVE_FINETUNED_SPECS:
            market_context = resolve_live_market_context(slug)
            data_path = market_context['path']
            market_source = market_context['source']
            market_source_label_value = market_context['source_label']

        try:
            if slug in LIVE_FINETUNED_SPECS:
                forecast_payload = build_live_finetuned_forecast(slug, data_path=data_path)
                data_path = forecast_payload['data_path']
                signal = forecast_payload['next_signal']
                prediction_path = forecast_payload['model_path']
                prediction_timestamp = signal['timestamp']
                forecast_rows = forecast_payload['forecast_rows']
                source_generated_at = forecast_payload['generated_at']
                status_label = LIVE_FINETUNED_SPECS[slug]['status_label']
            elif slug == '1day':
                forecast_payload = build_daily_live_forecast()
                if forecast_payload:
                    signal = forecast_payload['next_signal']
                    prediction_path = forecast_payload['path']
                    prediction_timestamp = signal['timestamp'] if signal else None
                    forecast_rows = forecast_payload['forecast_rows']
                    source_generated_at = forecast_payload.get('generated_at')
            else:
                model_df = get_model_frame(slug)
                latest_row = model_df.iloc[-1]
                signal = {
                    'timestamp': serialize_timestamp(latest_row['timestamp']),
                    'direction': 'Up' if int(latest_row['pred_dir']) == 1 else 'Down',
                    'confidence_pct': serialize_float(
                        float(latest_row['pred_prob_up']) * 100 if not pd.isna(latest_row.get('pred_prob_up')) and int(latest_row['pred_dir']) == 1 else
                        (1 - float(latest_row['pred_prob_up'])) * 100 if not pd.isna(latest_row.get('pred_prob_up')) else None
                    ),
                    'move_pct': None,
                    'predicted_close': serialize_float(latest_row.get('predicted_close')),
                    'actual_close': serialize_float(latest_row.get('actual_close')),
                }
                prediction_path = spec['prediction_path']
                prediction_timestamp = signal['timestamp']
                source_generated_at = file_mtime_iso(prediction_path)
        except Exception as exc:
            forecast_error = str(exc)

        market_payload = build_live_market_payload(data_path)

        cards.append({
            'slug': slug,
            'display_interval': spec['display_interval'],
            'title': spec['title'],
            'status_label': status_label,
            'market': market_payload,
            'signal': signal,
            'forecast_rows': forecast_rows,
            'source_generated_at': source_generated_at,
            'prediction_path': str(prediction_path) if prediction_path else None,
            'prediction_path_mtime': artifact_mtime_iso(prediction_path) if prediction_path else None,
            'data_path': str(data_path) if data_path else None,
            'data_path_mtime': file_mtime_iso(data_path) if data_path else None,
            'market_source': market_source,
            'market_source_label': market_source_label_value,
            'market_lag_minutes': minutes_since(market_payload['last']['timestamp']) if market_payload else None,
            'prediction_lag_minutes': minutes_since(prediction_timestamp),
            'error': combine_error_messages(market_context.get('error') if market_context else None, forecast_error),
        })

    return {
        'generated_at': current_utc_iso(),
        'poll_seconds': LIVE_POLL_SECONDS,
        'cards': cards,
    }

def load_data_files():
    """Scan data directory and return available data files"""
    data_files = []
    seen_paths = set()
    data_dirs = {
        BASE_DIR / 'data',
        FINETUNE_DIR / 'data',
    }
    for site_spec in SITE_TIMEFRAMES.values():
        data_dirs.add(Path(site_spec['actual_path']).parent)
        for candidate in site_spec.get('live_actual_candidates', []):
            data_dirs.add(Path(candidate).parent)

    for data_dir in sorted(data_dirs):
        if not data_dir.exists():
            continue
        for file in sorted(data_dir.iterdir()):
            if file.suffix not in {'.csv', '.feather'} or str(file) in seen_paths:
                continue
            file_size = file.stat().st_size
            data_files.append({
                'name': file.name,
                'path': str(file),
                'size': f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
            })
            seen_paths.add(str(file))

    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'])
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'])
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'])
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position, not always from the beginning of df
    if historical_start_idx + lookback + pred_len <= len(df):
        # Display lookback historical points + pred_len prediction points starting from specified position
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
        prediction_range = range(historical_start_idx+lookback, historical_start_idx+lookback+pred_len)
    else:
        # If data is insufficient, adjust to maximum available range
        available_lookback = min(lookback, len(df) - historical_start_idx)
        available_pred_len = min(pred_len, max(0, len(df) - historical_start_idx - available_lookback))
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
        prediction_range = range(historical_start_idx+available_lookback, historical_start_idx+available_lookback+available_pred_len)
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data (candlestick chart)
    fig.add_trace(go.Candlestick(
        x=historical_df['timestamps'] if 'timestamps' in historical_df.columns else historical_df.index,
        open=historical_df['open'],
        high=historical_df['high'],
        low=historical_df['low'],
        close=historical_df['close'],
        name='Historical Data (400 data points)',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add prediction data (candlestick chart)
    if pred_df is not None and len(pred_df) > 0:
        # Calculate prediction data timestamps - ensure continuity with historical data
        if 'timestamps' in df.columns and len(historical_df) > 0:
            # Start from the last timestamp of historical data, create prediction timestamps with the same time interval
            last_timestamp = historical_df['timestamps'].iloc[-1]
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
            
            pred_timestamps = pd.date_range(
                start=last_timestamp + time_diff,
                periods=len(pred_df),
                freq=time_diff
            )
        else:
            # If no timestamps, use index
            pred_timestamps = range(len(historical_df), len(historical_df) + len(pred_df))
        
        fig.add_trace(go.Candlestick(
            x=pred_timestamps,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name='Prediction Data (120 data points)',
            increasing_line_color='#66BB6A',
            decreasing_line_color='#FF7043'
        ))
    
    # Add actual data for comparison (if exists)
    if actual_df is not None and len(actual_df) > 0:
        # Actual data should be in the same time period as prediction data
        if 'timestamps' in df.columns:
            # Actual data should use the same timestamps as prediction data to ensure time alignment
            if 'pred_timestamps' in locals():
                actual_timestamps = pred_timestamps
            else:
                # If no prediction timestamps, calculate from the last timestamp of historical data
                if len(historical_df) > 0:
                    last_timestamp = historical_df['timestamps'].iloc[-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0] if len(df) > 1 else pd.Timedelta(hours=1)
                    actual_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=len(actual_df),
                        freq=time_diff
                    )
                else:
                    actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        else:
            actual_timestamps = range(len(historical_df), len(historical_df) + len(actual_df))
        
        fig.add_trace(go.Candlestick(
            x=actual_timestamps,
            open=actual_df['open'],
            high=actual_df['high'],
            low=actual_df['low'],
            close=actual_df['close'],
            name='Actual Data (120 data points)',
            increasing_line_color='#FF9800',
            decreasing_line_color='#F44336'
        ))
    
    # Update layout
    fig.update_layout(
        title='Kronos Financial Prediction Results - 400 Historical Points + 120 Prediction Points vs 120 Actual Points',
        xaxis_title='Time',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        showlegend=True
    )
    
    # Ensure x-axis time continuity
    if 'timestamps' in historical_df.columns:
        # Get all timestamps and sort them
        all_timestamps = []
        if len(historical_df) > 0:
            all_timestamps.extend(historical_df['timestamps'])
        if 'pred_timestamps' in locals():
            all_timestamps.extend(pred_timestamps)
        if 'actual_timestamps' in locals():
            all_timestamps.extend(actual_timestamps)
        
        if all_timestamps:
            all_timestamps = sorted(all_timestamps)
            fig.update_xaxes(
                range=[all_timestamps[0], all_timestamps[-1]],
                rangeslider_visible=False,
                type='date'
            )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/')
@app.route('/historical')
@app.route('/historical/overview')
def historical_overview():
    """Historical overview page."""
    context = build_overview_context()
    return render_template(
        'historical_overview.html',
        active_page='overview',
        page_title='Kronos Horizon Dashboard',
        **context,
    )


@app.route('/historical/detail')
def historical_detail():
    """Detailed historical analysis page."""
    selected_slug = normalize_timeframe(request.args.get('timeframe', '5min'))
    detail_payload = build_detail_payload(selected_slug)
    return render_template(
        'historical_detail.html',
        active_page='detail',
        page_title='Detailed Historical Analysis',
        models=build_site_summary(),
        selected_slug=selected_slug,
        detail_payload=detail_payload,
        generated_at=datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
    )


@app.route('/live')
def live_predictions():
    """Polling live page."""
    return render_template(
        'live.html',
        active_page='live',
        page_title='Live Polling Predictions',
        models=build_site_summary(),
        live_snapshot=build_live_snapshot(),
    )


@app.route('/lab')
def legacy_lab():
    """Legacy Kronos demo page."""
    return render_template('index.html')


@app.route('/api/live-snapshot')
def get_live_snapshot():
    """Return the live polling payload."""
    return jsonify(build_live_snapshot())

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor, loaded_model_key, loaded_model_info
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        loaded_model_key = model_key
        loaded_model_info = {
            'key': model_key,
            'name': model_config['name'],
            'params': model_config['params'],
            'context_length': model_config['context_length'],
            'description': model_config['description'],
        }
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'key': model_key,
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'key': loaded_model_key,
                    'name': loaded_model_info['name'] if loaded_model_info else predictor.model.__class__.__name__,
                    'params': loaded_model_info['params'] if loaded_model_info else None,
                    'context_length': loaded_model_info['context_length'] if loaded_model_info else None,
                    'description': loaded_model_info['description'] if loaded_model_info else None,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
