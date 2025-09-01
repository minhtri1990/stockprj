import streamlit as st
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
from pathlib import Path

from utils.trend_detect import TrendDetector
from utils.cached_stock_service import CachedStockService
from utils.stock_chart_builder_plotly import plot_chart 
from utils.load_all_symbols import get_symbols

st.set_page_config(page_title="VN Stock Viewer", page_icon="📈", layout="wide")

# ================================
# Helpers
# ================================
def ensure_ma_columns(df: pd.DataFrame, ma_list):
    if df is None or df.empty or not ma_list:
        return df
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            pass
        df = df.sort_values('time').reset_index(drop=True)
    if 'close' not in df.columns:
        return df
    for w in sorted(set(int(x) for x in ma_list)):
        target = f"MA_{w}"
        if target in df.columns:
            continue
        candidates = [
            f"MA{w}", f"MA_{w}",
            f"SMA{w}", f"SMA_{w}",
            f"ma{w}", f"ma_{w}",
            f"ema{w}", f"EMA{w}", f"EMA_{w}"
        ]  # fallback naming
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            df[target] = pd.to_numeric(df[found], errors='coerce')
        else:
            df[target] = df['close'].rolling(window=w, min_periods=w).mean()
    return df

@st.cache_data(show_spinner=True, ttl=30)
def load_prices(symbol: str, start: str, end: str, interval: str, ma_list):
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    max_w = (max(ma_list) + max(ma_list)) if ma_list else 0
    if max_w > 0:
        if interval == "1D":
            fetch_start_dt = start_dt - timedelta(days=max_w)
        elif interval == "1W":
            fetch_start_dt = start_dt - timedelta(weeks=max_w)
        elif interval == "1M":
            fetch_start_dt = start_dt - timedelta(days=max_w * 30)
        else:
            fetch_start_dt = start_dt - timedelta(days=max_w)
    else:
        fetch_start_dt = start_dt

    service = CachedStockService(symbol)
    df = service.get_price_history(
        start=fetch_start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        add_ma=ma_list if ma_list else None
    )
    if df is None or df.empty:
        return df
    if 'time' in df.columns:
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            pass
    df = ensure_ma_columns(df, ma_list)
    mask = (df['time'].dt.date >= start_dt) & (df['time'].dt.date <= end_dt)
    df = df.loc[mask].reset_index(drop=True)
    return df


# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("Thiết lập")
    symbols = get_symbols()
    default_idx = 0
    symbol = st.selectbox("Mã cổ phiếu", symbols, index=0)

    today = date.today()
    default_start = today - timedelta(days=365*2)
    start_date = st.date_input("Từ ngày", default_start)
    end_date = st.date_input("Đến ngày", today)
    interval = st.selectbox("Interval", ["1D", "1W", "1M"], index=0)
    ma_options = st.multiselect("Đường MA", [5, 10, 20, 50, 100, 200], default=[10, 20, 50, 200])
    show_volume = st.checkbox("Hiển thị Volume", value=True)

    st.divider()
    st.subheader("Trend Detector")
    trend_method = st.selectbox("Phương pháp", ["multi", "ema", "adx", "donchian"], index=0)
    ema_fast = st.number_input("EMA Fast", 3, 200, 12)
    ema_slow = st.number_input("EMA Slow", 5, 300, 34)
    ema_slope_lookback = st.number_input("EMA slope lookback", 1, 50, 5)
    adx_period = st.number_input("ADX period", 5, 50, 14)
    adx_threshold = st.number_input("ADX threshold", 5, 60, 20)
    donchian_window = st.number_input("Donchian window", 5, 200, 20)
    donchian_proximity = st.slider("Donchian proximity", 0.01, 0.5, 0.1, 0.01)
    score_up_threshold = st.number_input("Ngưỡng UP (≥)", -10, 30, 3)
    score_down_threshold = st.number_input("Ngưỡng DOWN (≤)", -30, 10, -2)
    use_obv = st.checkbox("Thêm OBV slope (multi)", value=False)
    obv_lookback = st.number_input("OBV slope lookback", 1, 50, 5)
    keep_components = st.checkbox("Giữ cột chi tiết trend", value=False)
    compute_streak = st.checkbox("Tính streak & longest", value=True)

# ================================
# Kiểm tra ngày
# ================================
if start_date > end_date:
    st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
    st.stop()

# ================================
# Load giá
# ================================
with st.spinner("Đang tải dữ liệu..."):
    df_price = load_prices(
        symbol=symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval=interval,
        ma_list=ma_options
    )

if df_price is None or df_price.empty:
    st.warning("Không có dữ liệu.")
    st.stop()

plot_chart(df_price, symbol, interval, ma_options, show_volume=show_volume)

# ================================
# Trend detect
# ================================
detector = TrendDetector(
    method=trend_method,
    ema_fast=ema_fast,
    ema_slow=ema_slow,
    ema_slope_lookback=ema_slope_lookback,
    adx_period=adx_period,
    adx_threshold=adx_threshold,
    donchian_window=donchian_window,
    donchian_proximity=donchian_proximity,
    score_up_threshold=score_up_threshold,
    score_down_threshold=score_down_threshold,
    use_obv=(use_obv and trend_method == "multi"),
    obv_lookback=obv_lookback,
    keep_components=keep_components,
    compute_streak=compute_streak
)
df_trend = detector.detect(df_price)
stats = detector.get_stats()
score_col = stats.get("score_col")
label_col = stats.get("label_col")
streak_col = stats.get("streak_col")
current_trend = stats.get("current_trend")
current_streak = stats.get("current_streak_len")

if len(df_trend) > 1:
    prev_score = df_trend.iloc[-2][score_col]
    delta_score = df_trend.iloc[-1][score_col] - prev_score
else:
    delta_score = np.nan

cols = st.columns(4 if compute_streak else 3)
cols[0].metric("Trend hiện tại", current_trend)
cols[1].metric("Score", f"{df_trend.iloc[-1][score_col]}")
cols[2].metric("Δ Score", f"{delta_score:+.0f}" if pd.notna(delta_score) else "N/A")
if compute_streak:
    cols[3].metric(
        "Chuỗi hiện tại",
        f"{current_streak} phiên" if current_streak else "—",
        help="Số phiên liên tiếp cùng nhãn trend."
    )

base_show = ["time", "close", score_col, label_col]
if compute_streak and streak_col in df_trend.columns:
    base_show.append(streak_col)

with st.expander("10 phiên cuối (trend)"):
    st.dataframe(df_trend[base_show].tail(10).sort_values("time", ascending=False), width='stretch')

with st.expander("Toàn bộ bảng trend / tải CSV / thống kê"):
    if "time" in df_trend.columns:
        try:
            df_trend["time"] = pd.to_datetime(df_trend["time"])
        except Exception:
            pass
        df_view = df_trend[base_show].sort_values("time", ascending=False)
    else:
        df_view = df_trend[base_show]
    st.dataframe(df_view, width='stretch')
    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("Tải CSV", data=csv, file_name=f"{symbol}_trend.csv", mime="text/csv")
    if compute_streak:
        st.write("Longest per label:", {k: int(v) for k, v in stats.get("longest_per_label", {}).items()})
        st.write("Counts:", stats.get("counts"))

# ================================
# Debug
# ================================
with st.expander("Hiện debug cột giá"):
    st.write(df_price.columns.tolist())
    st.write("MA columns:", [c for c in df_price.columns if c.startswith("MA_")])