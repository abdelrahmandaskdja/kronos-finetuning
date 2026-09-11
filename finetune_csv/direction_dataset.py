import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DirectionKlineDataset(Dataset):
    """Single-asset direction dataset for Kronos direction finetuning.

    `horizon_steps` defines the prediction horizon on bar data:
    1 for 5m, 12 for 1h, and 288 for 1d in 24/7 markets.

    Ambiguous moves are masked by a local threshold:
    eps_t = max(vol_k * sigma_t, min_eps),
    where sigma_t is rolling std of horizon returns over `vol_window` bars.
    If |ret| <= eps_t, label is ignored via `mask = 0`.
    Set `mask_ambiguous=False` to train/evaluate on every finite horizon return.
    """

    def __init__(
        self,
        data_path: str,
        data_type: str = "train",
        lookback_window: int = 400,
        horizon_steps: int = 1,
        clip: float = 5.0,
        vol_window: int = 288,
        vol_k: float = 0.25,
        min_eps: float = 0.0,
        mask_ambiguous: bool = True,
        seed: int = 100,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if lookback_window <= 0:
            raise ValueError("lookback_window must be > 0.")
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be > 0.")
        if vol_window <= 0:
            raise ValueError("vol_window must be > 0.")
        if vol_k < 0:
            raise ValueError("vol_k must be >= 0.")
        if min_eps < 0:
            raise ValueError("min_eps must be >= 0.")

        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.horizon_steps = horizon_steps
        self.window = lookback_window + horizon_steps
        self.clip = clip
        self.vol_window = vol_window
        self.vol_k = vol_k
        self.min_eps = min_eps
        self.mask_ambiguous = bool(mask_ambiguous)
        self.seed = seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.feature_list = ["open", "high", "low", "close", "volume", "amount"]
        self.time_feature_list = ["minute", "hour", "weekday", "day", "month"]

        self.py_rng = random.Random(seed)

        self._load_and_preprocess_data()
        self._split_data_by_time()
        self._build_horizon_stats()

        self.n_samples = len(self.data) - self.window + 1
        if self.n_samples <= 0:
            raise ValueError(
                "Data length after split is insufficient for lookback_window + horizon_steps."
            )

    def _load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)

        required_cols = ["timestamps"] + self.feature_list
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        df["timestamps"] = pd.to_datetime(df["timestamps"])
        df = df.sort_values("timestamps").reset_index(drop=True)

        df["minute"] = df["timestamps"].dt.minute
        df["hour"] = df["timestamps"].dt.hour
        df["weekday"] = df["timestamps"].dt.weekday
        df["day"] = df["timestamps"].dt.day
        df["month"] = df["timestamps"].dt.month

        cols = self.feature_list + self.time_feature_list
        if df[cols].isnull().any().any():
            df[cols] = df[cols].ffill()

        self.data = df[cols].copy()
        self.timestamps = df["timestamps"].copy()

    def _split_data_by_time(self):
        total_length = len(self.data)
        train_end = int(total_length * self.train_ratio)
        val_end = int(total_length * (self.train_ratio + self.val_ratio))

        if self.data_type == "train":
            self.data = self.data.iloc[:train_end].copy()
            self.timestamps = self.timestamps.iloc[:train_end].copy()
        elif self.data_type == "val":
            self.data = self.data.iloc[train_end:val_end].copy()
            self.timestamps = self.timestamps.iloc[train_end:val_end].copy()
        elif self.data_type == "test":
            self.data = self.data.iloc[val_end:].copy()
            self.timestamps = self.timestamps.iloc[val_end:].copy()
        else:
            raise ValueError("data_type must be one of: 'train', 'val', 'test'.")

        self.data = self.data.reset_index(drop=True)
        self.timestamps = self.timestamps.reset_index(drop=True)

    def _build_horizon_stats(self):
        close = self.data["close"].to_numpy(dtype=np.float64)
        n = len(close)

        horizon_ret = np.full(n, np.nan, dtype=np.float64)
        if n > self.horizon_steps:
            with np.errstate(divide="ignore", invalid="ignore"):
                horizon_ret[: n - self.horizon_steps] = np.log(
                    close[self.horizon_steps :] / close[: n - self.horizon_steps]
                )

        sigma = (
            pd.Series(horizon_ret)
            .rolling(window=self.vol_window, min_periods=1)
            .std(ddof=0)
            .to_numpy()
        )
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)

        # Keep per-row horizon return and rolling volatility in the dataframe
        # for downstream tasks (e.g., regression on horizon return).
        self.data["ret_horizon"] = horizon_ret
        self.data["vol_horizon"] = sigma

        self.horizon_ret = horizon_ret
        self.horizon_vol = sigma
        self.local_eps = np.maximum(self.vol_k * sigma, self.min_eps)

    def set_epoch_seed(self, epoch: int):
        """Reseed internal RNG per epoch, mirroring Kronos dataset interfaces."""
        epoch_seed = self.seed + int(epoch)
        self.py_rng.seed(epoch_seed)
        self.current_epoch = int(epoch)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(f"idx out of range: {idx} (valid: 0 to {self.n_samples - 1})")

        start_idx = idx
        end_idx = start_idx + self.window
        window_df = self.data.iloc[start_idx:end_idx]

        hist_df = window_df.iloc[: self.lookback_window]

        x = hist_df[self.feature_list].values.astype(np.float32)
        x_stamp = hist_df[self.time_feature_list].values.astype(np.float32)

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip).astype(np.float32)

        anchor = start_idx + self.lookback_window - 1
        ret = float(self.data["ret_horizon"].iloc[anchor])
        vol = float(self.data["vol_horizon"].iloc[anchor])

        # Direction label + mask.
        # In masked mode, small or invalid moves are skipped.
        # In unmasked mode, every finite horizon return is scored.
        if not np.isfinite(ret):
            ret_value = 0.0
            label = 0.0
            mask = 0.0
        else:
            ret_value = float(ret)
            label = 1.0 if ret > 0.0 else 0.0

            if not self.mask_ambiguous:
                mask = 1.0
            else:
                if not np.isfinite(vol) or vol <= 0.0:
                    mask = 0.0
                else:
                    eps_t = max(self.vol_k * vol, self.min_eps)
                    if abs(ret) <= eps_t:
                        mask = 0.0
                    else:
                        mask = 1.0

        return {
            "x": torch.from_numpy(x),
            "x_stamp": torch.from_numpy(x_stamp),
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "ret": torch.tensor(ret_value, dtype=torch.float32),
        }
