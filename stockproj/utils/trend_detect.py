# ================================
# File: stockproj/trend_detect.py
# ================================
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any

TrendMethod = Literal["ema", "adx", "donchian", "multi"]

@dataclass
class TrendConfig:
    method: TrendMethod = "multi"
    ema_fast: int = 12
    ema_slow: int = 34
    ema_slope_lookback: int = 5
    adx_period: int = 14
    adx_threshold: float = 20.0
    donchian_window: int = 20
    donchian_proximity: float = 0.10
    score_up_threshold: int = 3
    score_down_threshold: int = -2
    use_obv: bool = False
    obv_lookback: int = 5
    keep_components: bool = True
    prefix: str = ""
    # --- MỚI: tính chuỗi liên tiếp ---
    compute_streak: bool = True
    streak_col_name: str = "trend_streak_len"

class TrendDetector:
    """
    Phát hiện xu hướng với nhiều phương pháp:
      - ema
      - adx
      - donchian
      - multi (tổng hợp)
    Có thể thêm OBV slope nếu có cột volume.
    Nay bổ sung thống kê streak (chuỗi trend liên tiếp).
    """
    def __init__(self, config: Optional[TrendConfig] = None, **kwargs):
        if config is None:
            config = TrendConfig()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.config = config
        self.last_stats: Dict[str, Any] = {}

    # ------------ Helpers ------------
    @staticmethod
    def ensure_ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
        name = f"EMA_{period}"
        if name not in df.columns:
            df[name] = df[col].ewm(span=period, adjust=False).mean()
        return df[name]

    @staticmethod
    def ema_slope(series: pd.Series, lookback: int) -> pd.Series:
        return series - series.shift(lookback)

    @staticmethod
    def calc_adx(df: pd.DataFrame, period: int = 14,
                 high_col: str = "high", low_col: str = "low", close_col: str = "close") -> pd.DataFrame:
        if {high_col, low_col, close_col}.difference(df.columns):
            raise ValueError("Thiếu cột high/low/close để tính ADX.")
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = high - low
        prev_close = close.shift()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = np.max(np.vstack([tr1.values, tr2.values, tr3.values]), axis=0)

        tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
        minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["adx"] = adx
        return df

    @staticmethod
    def calc_donchian(df: pd.DataFrame, window: int,
                      high_col: str = "high", low_col: str = "low") -> pd.DataFrame:
        df[f"donchian_high_{window}"] = df[high_col].rolling(window).max()
        df[f"donchian_low_{window}"] = df[low_col].rolling(window).min()
        return df

    @staticmethod
    def calc_obv(df: pd.DataFrame, close_col: str = "close", vol_col: str = "volume") -> pd.Series:
        if vol_col not in df.columns:
            raise ValueError("Thiếu cột volume để tính OBV.")
        direction = np.sign(df[close_col].diff()).fillna(0)
        obv = (direction * df[vol_col]).cumsum()
        df["obv"] = obv
        return df["obv"]

    def _p(self, name: str) -> str:
        return f"{self.config.prefix}{name}" if self.config.prefix else name

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        self.last_stats = {}
        if df is None or df.empty:
            return df
        df = df.copy()

        # EMA part
        if cfg.method in ("ema", "multi"):
            fast = self.ensure_ema(df, cfg.ema_fast)
            slow = self.ensure_ema(df, cfg.ema_slow)
            df[self._p("ema_fast_gt_slow")] = (fast > slow).astype(int)
            df[self._p("ema_slow_slope")] = self.ema_slope(slow, cfg.ema_slope_lookback)
            df[self._p("ema_slow_slope_dir")] = np.select(
                [df[self._p("ema_slow_slope")] > 0, df[self._p("ema_slow_slope")] < 0],
                [1, -1], default=0
            )
            df[self._p("close_above_fast")] = (df["close"] > fast).astype(int)

        # ADX part
        if cfg.method in ("adx", "multi"):
            df = self.calc_adx(df, cfg.adx_period)
            df[self._p("adx_strong")] = (df["adx"] > cfg.adx_threshold).astype(int)
            df[self._p("di_bias")] = np.select(
                [df["plus_di"] > df["minus_di"], df["plus_di"] < df["minus_di"]],
                [1, -1], default=0
            )

        # Donchian part
        if cfg.method in ("donchian", "multi"):
            df = self.calc_donchian(df, cfg.donchian_window)
            upper = df[f"donchian_high_{cfg.donchian_window}"]
            lower = df[f"donchian_low_{cfg.donchian_window}"]
            rng = (upper - lower).replace(0, np.nan)
            mid = (upper + lower) / 2
            df[self._p("donchian_pos")] = (df["close"] - lower) / rng
            df[self._p("near_upper")] = ((upper - df["close"]) <= cfg.donchian_proximity * (upper - lower)).astype(int)
            df[self._p("near_lower")] = ((df["close"] - lower) <= cfg.donchian_proximity * (upper - lower)).astype(int)
            df[self._p("donchian_bias")] = np.select(
                [df["close"] > mid, df["close"] < mid],
                [1, -1], default=0
            )

        # OBV part
        if cfg.use_obv:
            if "volume" in df.columns:
                self.calc_obv(df)
                df[self._p("obv_slope")] = df["obv"] - df["obv"].shift(cfg.obv_lookback)
                df[self._p("obv_slope_dir")] = np.select(
                    [df[self._p("obv_slope")] > 0, df[self._p("obv_slope")] < 0],
                    [1, -1], default=0
                )
            else:
                df[self._p("obv_slope_dir")] = 0
        else:
            df[self._p("obv_slope_dir")] = 0

        score_col = self._p("trend_score")
        df[score_col] = 0

        if cfg.method == "ema":
            df[score_col] += df.get(self._p("ema_fast_gt_slow"), 0)
            df[score_col] += df.get(self._p("close_above_fast"), 0)
            df[score_col] += df.get(self._p("ema_slow_slope_dir"), 0)
        elif cfg.method == "adx":
            df[score_col] += df.get(self._p("di_bias"), 0)
            df[score_col] += df.get(self._p("adx_strong"), 0)
        elif cfg.method == "donchian":
            df[score_col] += df.get(self._p("donchian_bias"), 0)
            df[score_col] += df.get(self._p("near_upper"), 0)
            df[score_col] -= df.get(self._p("near_lower"), 0)
        elif cfg.method == "multi":
            df[score_col] += df.get(self._p("ema_fast_gt_slow"), 0)
            df[score_col] += df.get(self._p("close_above_fast"), 0)
            df[score_col] += df.get(self._p("ema_slow_slope_dir"), 0)
            df[score_col] += df.get(self._p("di_bias"), 0)
            df[score_col] += df.get(self._p("adx_strong"), 0)
            df[score_col] += df.get(self._p("donchian_bias"), 0)
            df[score_col] += df.get(self._p("near_upper"), 0)
            df[score_col] -= df.get(self._p("near_lower"), 0)
            if cfg.use_obv:
                df[score_col] += df.get(self._p("obv_slope_dir"), 0)

        label_col = self._p("trend_label")
        df[label_col] = np.select(
            [df[score_col] >= cfg.score_up_threshold,
             df[score_col] <= cfg.score_down_threshold],
            ["UP", "DOWN"],
            default="SIDE"
        )

        # --- MỚI: tính streak & longest chain ---
        streak_col = self._p(cfg.streak_col_name)
        longest_per_label: Dict[str, int] = {}
        current_streak_len = None
        current_trend = None

        if cfg.compute_streak and label_col in df.columns:
            change_grp = (df[label_col] != df[label_col].shift()).cumsum()
            df[streak_col] = df.groupby(change_grp).cumcount() + 1
            # Lấy chuỗi dài nhất cho từng nhãn
            run_lengths = (
                df.assign(_grp=change_grp)
                  .groupby(['_grp', label_col])
                  .size()
                  .reset_index(name='run_length')
            )
            longest_per_label = (
                run_lengths.groupby(label_col)['run_length']
                .max()
                .to_dict()
            )
            current_trend = df.iloc[-1][label_col]
            current_streak_len = int(df.iloc[-1][streak_col])
        else:
            # Nếu không compute streak, đảm bảo cột không tồn tại hoặc giữ cũ
            if streak_col in df.columns and not cfg.keep_components:
                df.drop(columns=[streak_col], inplace=True, errors="ignore")

        counts = df[label_col].value_counts().to_dict()

          # Lưu thống kê
        self.last_stats = {
            "current_trend": current_trend if current_trend is not None else (df.iloc[-1][label_col] if len(df) else None),
            "current_streak_len": current_streak_len,
            "counts": counts,
            "longest_per_label": longest_per_label,
            "score_col": score_col,
            "label_col": label_col,
            "streak_col": streak_col if cfg.compute_streak else None
        }

        if not cfg.keep_components:
            base_cols = ["time", "open", "high", "low", "close"]
            if "volume" in df.columns:
                base_cols.append("volume")
            extra_cols = [score_col, label_col]
            if cfg.compute_streak and streak_col in df.columns:
                extra_cols.append(streak_col)
            return df[[c for c in base_cols + extra_cols if c in df.columns]]

        return df

    def get_params(self) -> Dict[str, Any]:
        return asdict(self.config)

    def get_stats(self) -> Dict[str, Any]:
        return self.last_stats.copy()

def detect_trend(df: pd.DataFrame, method: TrendMethod = "multi", **kwargs) -> pd.DataFrame:
    """
    Hàm wrapper để gọi nhanh:
        df_out = detect_trend(df, method="ema", ema_fast=10, ema_slow=30)
    Các kwargs sẽ override cấu hình TrendConfig tương ứng.
    """
    config = TrendConfig(method=method)
    detector = TrendDetector(config=config, **kwargs)
    return detector.detect(df)

__all__ = ["TrendConfig", "TrendDetector", "detect_trend"]
