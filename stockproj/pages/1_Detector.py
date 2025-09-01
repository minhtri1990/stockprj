import os
import time
import glob
import numpy as np
import streamlit as st
from datetime import date, timedelta, datetime
from openpyxl import Workbook
from pathlib import Path
from PIL import Image
import plotly.io as pio
import pandas as pd
from ultralytics import YOLO

from utils.trend_detect import TrendDetector
from utils.cached_stock_service import CachedStockService
from utils.load_all_symbols import get_symbols

from utils.stock_chart_builder_plotly import builder_chart
from indicator.ART import plot_ATR
from indicator.MACD import plot_MACD
from indicator.RSI import plot_RSI
from indicator.RSI_K_D import plot_stoch_RSI

# ================================
# Setup directories
# ================================
save_path = "./"
screenshots_path = os.path.join(save_path, "runs/screenshots/")
detect_path = os.path.join(save_path, "runs/detect/")
os.makedirs(save_path, exist_ok=True)
os.makedirs(screenshots_path, exist_ok=True)

# ================================
# Define pattern classes
# ================================
#classes = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'StockLine', 'Triangle', 'W_Bottom']
classes = ['Down', 'Up']
model_path = "D:/Projects/AI project/stockproj/stockproj/pages/model_2.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = YOLO(model_path)
# set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# ================================
# Streamlit Setup
# ================================
st.set_page_config(page_title="VN Stock Viewer", page_icon="📈", layout="wide")
st.title("📈 Biểu đồ")

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

# ================================
fig_bar_chart = builder_chart(df_price, symbol, interval, ma_options, show_volume=show_volume)
st.plotly_chart(fig_bar_chart, width='stretch')
# Plot ATR chart
fig_plot_ATR = plot_ATR(df_price)
st.plotly_chart(fig_plot_ATR, width='stretch')
# Plot RSI chart
fig_RSI = plot_RSI(df_price)
st.plotly_chart(fig_RSI, width='stretch')
# Plot MACD chart
fig_MACD = plot_MACD(df_price)
st.plotly_chart(fig_MACD, width='stretch')
# Plot Stochastic RSI chart
fig_stoch_RSI = plot_stoch_RSI(df_price)
st.plotly_chart(fig_stoch_RSI, width='stretch')

