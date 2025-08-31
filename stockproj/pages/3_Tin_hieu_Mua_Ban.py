import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List
import plotly.graph_objects as go

from utils.cached_stock_service import CachedStockService
from utils.load_all_symbols import load_all_symbols
# TrendDetector là tùy chọn
try:
    from utils.trend_detect import TrendDetector
    HAS_TREND_DETECTOR = True
except Exception:
    HAS_TREND_DETECTOR = False

st.set_page_config(page_title="Tín hiệu Mua / Bán", page_icon="🟢", layout="wide")
st.title("🟢🔴 Trang tín hiệu Mua / Bán cổ phiếu")

# ---------------- Cache ----------------
@st.cache_data(ttl=600, show_spinner=False)
def get_symbols():
    return load_all_symbols()

@st.cache_data(ttl=60, show_spinner=True)
def load_price_history(symbol: str, start: str, end: str, interval: str):
    svc = CachedStockService(symbol)
    df = svc.get_price_history(start=start, end=end, interval=interval)
    if df is None or df.empty:
        return pd.DataFrame()
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass
        df = df.sort_values("time").reset_index(drop=True)
    return df

# ---------------- Indicators ----------------
def ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    # Wilder smoothing fallback
    avg_gain = avg_gain.combine_first(gain.ewm(alpha=1/period, adjust=False).mean())
    avg_loss = avg_loss.combine_first(loss.ewm(alpha=1/period, adjust=False).mean())
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14):
    h = df["high"]; l = df["low"]; c = df["close"]; prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def donchian(df: pd.DataFrame, window: int = 20):
    return df["high"].rolling(window).max(), df["low"].rolling(window).min()

# ---------------- Signal generator ----------------
def generate_signals(
    df: pd.DataFrame,
    fast_ma: int,
    slow_ma: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    rsi_period: int,
    rsi_buy_th: int,
    rsi_sell_th: int,
    donchian_window: int,
    atr_period: int,
    buy_score_threshold: int,
    sell_score_threshold: int,
    show_trend: bool,
    trend_method: str
):
    if df.empty:
        return df, pd.DataFrame()

    df = df.copy()
    # Core indicators
    df["EMA_fast"] = ema(df["close"], fast_ma)
    df["EMA_slow"] = ema(df["close"], slow_ma)
    df["MACD_fast"] = ema(df["close"], macd_fast)
    df["MACD_slow"] = ema(df["close"], macd_slow)
    df["MACD"] = df["MACD_fast"] - df["MACD_slow"]
    df["MACD_signal"] = ema(df["MACD"], macd_signal)
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["RSI"] = rsi(df["close"], rsi_period)
    df["ATR"] = atr(df, atr_period)
    dc_high, dc_low = donchian(df, donchian_window)
    df["DC_high"], df["DC_low"] = dc_high, dc_low

    # Cross detections
    def cross_up(s1, s2): return (s1 > s2) & (s1.shift(1) <= s2.shift(1))
    def cross_down(s1, s2): return (s1 < s2) & (s1.shift(1) >= s2.shift(1))
    df["sig_ma_buy"] = cross_up(df["EMA_fast"], df["EMA_slow"])
    df["sig_ma_sell"] = cross_down(df["EMA_fast"], df["EMA_slow"])
    df["sig_macd_buy"] = cross_up(df["MACD"], df["MACD_signal"])
    df["sig_macd_sell"] = cross_down(df["MACD"], df["MACD_signal"])
    df["sig_rsi_buy"] = (df["RSI"] > rsi_buy_th) & (df["RSI"].shift(1) <= rsi_buy_th)
    df["sig_rsi_sell"] = (df["RSI"] < rsi_sell_th) & (df["RSI"].shift(1) >= rsi_sell_th)
    prev_high = df["high"].rolling(donchian_window).max().shift(1)
    prev_low = df["low"].rolling(donchian_window).min().shift(1)
    df["sig_dc_buy"] = df["close"] > prev_high
    df["sig_dc_sell"] = df["close"] < prev_low

    # Trend ONLY for display (no filtering)
    if show_trend and HAS_TREND_DETECTOR:
        try:
            detector = TrendDetector(method=trend_method)
            df_td = detector.detect(df[["time","open","high","low","close","volume"]].copy())
            stats = detector.get_stats()
            label_col = stats.get("label_col")
            if label_col in df_td.columns:
                df = df.merge(df_td[["time", label_col]], on="time", how="left")
                df["trend_label"] = df[label_col]
            else:
                df["trend_label"] = np.nan
        except Exception as e:
            st.warning(f"Lỗi TrendDetector (hiển thị): {e}")
            df["trend_label"] = np.nan
    else:
        df["trend_label"] = np.nan

    # Aggregate (unchanged by trend)
    buy_components = ["sig_ma_buy","sig_macd_buy","sig_rsi_buy","sig_dc_buy"]
    sell_components = ["sig_ma_sell","sig_macd_sell","sig_rsi_sell","sig_dc_sell"]
    df["buy_score_raw"] = df[buy_components].sum(axis=1)
    df["sell_score_raw"] = df[sell_components].sum(axis=1)
    df["buy_signal_final"] = df["buy_score_raw"] >= buy_score_threshold
    df["sell_signal_final"] = df["sell_score_raw"] >= sell_score_threshold

    # Risk management suggestion
    rr = st.session_state.get("risk_reward_ratio", 2.0)
    atr_mult = st.session_state.get("atr_multiple", 1.5)
    df["suggest_stop"] = np.where(
        df["buy_signal_final"],
        df["close"] - df["ATR"] * atr_mult,
        np.where(df["sell_signal_final"], df["close"] + df["ATR"] * atr_mult, np.nan)
    )
    df["suggest_tp"] = np.where(
        df["buy_signal_final"],
        df["close"] + (df["close"] - df["suggest_stop"]) * rr,
        np.where(df["sell_signal_final"], df["close"] - (df["suggest_stop"] - df["close"]) * rr, np.nan)
    )

    # Events table
    events = []
    for _, row in df.iterrows():
        if row["buy_signal_final"] or row["sell_signal_final"]:
            side = "BUY" if row["buy_signal_final"] else "SELL"
            reasons = []
            if row["sig_ma_buy"] or row["sig_ma_sell"]: reasons.append("MA cross")
            if row["sig_macd_buy"] or row["sig_macd_sell"]: reasons.append("MACD cross")
            if row["sig_rsi_buy"] or row["sig_rsi_sell"]: reasons.append("RSI cross")
            if row["sig_dc_buy"] or row["sig_dc_sell"]: reasons.append("Donchian breakout")
            # Trend chỉ bổ sung thông tin
            trlab = row.get("trend_label")
            events.append({
                "time": row["time"],
                "side": side,
                "trend": trlab,
                "close": row["close"],
                "buy_score": row["buy_score_raw"],
                "sell_score": row["sell_score_raw"],
                "reasons": ", ".join(reasons),
                "stop": row["suggest_stop"],
                "take_profit": row["suggest_tp"]
            })
    df_events = pd.DataFrame(events)
    return df, df_events

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Cấu hình")
    symbols = get_symbols()
    default_idx = 0
    for cand in ["HPG", "VCB", "FPT"]:
        if cand in symbols:
            default_idx = symbols.index(cand)
            break
    symbol = st.selectbox("Mã", symbols, index=default_idx)

    today = date.today()
    start_date = st.date_input("Từ ngày", today - timedelta(days=365))
    end_date = st.date_input("Đến ngày", today)
    interval = st.selectbox("Interval", ["1D", "1W", "1M"], index=0)

    st.subheader("MA & MACD")
    fast_ma = st.number_input("EMA nhanh", 3, 100, 10)
    slow_ma = st.number_input("EMA chậm", 5, 300, 20)
    macd_fast = st.number_input("MACD fast", 3, 50, 12)
    macd_slow = st.number_input("MACD slow", 5, 100, 26)
    macd_signal = st.number_input("MACD signal", 3, 50, 9)

    st.subheader("RSI")
    rsi_period = st.number_input("RSI period", 5, 50, 14)
    rsi_buy_th = st.number_input("RSI hồi lên qua ngưỡng (ví dụ 30)", 5, 60, 30)
    rsi_sell_th = st.number_input("RSI giảm xuống qua ngưỡng (ví dụ 70)", 40, 95, 70)

    st.subheader("Donchian & ATR")
    donchian_window = st.number_input("Donchian window", 5, 120, 20)
    atr_period = st.number_input("ATR period", 5, 50, 14)
    atr_multiple = st.number_input("ATR multiple (Stop)", 0.5, 10.0, 1.5, 0.1, key="atr_multiple")
    risk_reward_ratio = st.number_input("Risk:Reward", 0.5, 10.0, 2.0, 0.1, key="risk_reward_ratio")

    st.subheader("Ngưỡng tổng hợp")
    buy_score_threshold = st.number_input("Ngưỡng BUY score (>=)", 1, 5, 2)
    sell_score_threshold = st.number_input("Ngưỡng SELL score (>=)", 1, 5, 2)

    st.subheader("Trend (chỉ hiển thị)")
    show_trend = st.checkbox("Tính & hiển thị Trend (không lọc tín hiệu)", value=True and HAS_TREND_DETECTOR, disabled=not HAS_TREND_DETECTOR)
    trend_method = st.selectbox("Trend method", ["multi", "ema", "adx", "donchian"], index=0, disabled=not show_trend or not HAS_TREND_DETECTOR)
    show_trend_on_markers = st.checkbox("Hiện chữ trend trên marker", value=True)

    show_debug = st.checkbox("Debug indicators", value=False)

if start_date > end_date:
    st.error("Ngày bắt đầu phải <= ngày kết thúc.")
    st.stop()

# ---------------- Main flow ----------------
with st.spinner("Đang tải & tính toán..."):
    df_price = load_price_history(symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), interval)
    if df_price.empty:
        st.warning("Không có dữ liệu giá.")
        st.stop()
    df_ind, df_events = generate_signals(
        df_price,
        fast_ma, slow_ma,
        macd_fast, macd_slow, macd_signal,
        rsi_period, rsi_buy_th, rsi_sell_th,
        donchian_window,
        atr_period,
        buy_score_threshold,
        sell_score_threshold,
        show_trend,
        trend_method
    )

if df_events.empty:
    st.info("Không phát sinh tín hiệu theo cấu hình hiện tại.")
else:
    st.subheader("Bảng tín hiệu mới nhất")
    prefer_cols = ["time","side","trend","close","buy_score","sell_score","reasons","stop","take_profit"]
    cols_exist = [c for c in prefer_cols if c in df_events.columns]
    st.dataframe(df_events[cols_exist].sort_values("time", ascending=False).head(50), width='stretch')
    csv = df_events.to_csv(index=False).encode("utf-8")
    st.download_button("Tải CSV tín hiệu", csv, file_name=f"{symbol}_signals.csv", mime="text/csv")

st.subheader("Biểu đồ giá & tín hiệu")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_ind["time"],
    open=df_ind["open"],
    high=df_ind["high"],
    low=df_ind["low"],
    close=df_ind["close"],
    name="Price"
))
fig.add_trace(go.Scatter(
    x=df_ind["time"], y=df_ind["EMA_fast"],
    line=dict(color="orange", width=1),
    name=f"EMA{fast_ma}"
))
fig.add_trace(go.Scatter(
    x=df_ind["time"], y=df_ind["EMA_slow"],
    line=dict(color="blue", width=1),
    name=f"EMA{slow_ma}"
))

buys = df_ind[df_ind["buy_signal_final"]]
sells = df_ind[df_ind["sell_signal_final"]]
if not buys.empty:
    buy_text = buys["trend_label"] if show_trend_on_markers and "trend_label" in buys.columns else None
    fig.add_trace(go.Scatter(
        x=buys["time"], y=buys["close"],
        mode="markers+text" if buy_text is not None else "markers",
        text=buy_text if buy_text is not None else None,
        textposition="top center",
        marker=dict(symbol="triangle-up", size=12, color="green"),
        name="BUY"
    ))
if not sells.empty:
    sell_text = sells["trend_label"] if show_trend_on_markers and "trend_label" in sells.columns else None
    fig.add_trace(go.Scatter(
        x=sells["time"], y=sells["close"],
        mode="markers+text" if sell_text is not None else "markers",
        text=sell_text if sell_text is not None else None,
        textposition="bottom center",
        marker=dict(symbol="triangle-down", size=12, color="red"),
        name="SELL"
    ))

fig.update_layout(
    height=700,
    xaxis_title="Time",
    yaxis_title="Price",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_rangeslider_visible=False
)
st.plotly_chart(fig, width='stretch')

with st.expander("Indicators chi tiết (RSI, MACD, Donchian, ATR)"):
    st.write("RSI")
    st.line_chart(df_ind.set_index("time")["RSI"])

    st.write("MACD & Signal")
    macd_df = df_ind[["time","MACD","MACD_signal","MACD_hist"]].set_index("time")
    st.line_chart(macd_df[["MACD","MACD_signal"]])
    st.bar_chart(macd_df["MACD_hist"])

    st.write("Donchian High / Low")
    don_df = df_ind[["time","close","DC_high","DC_low"]].set_index("time")
    st.line_chart(don_df)

    st.write("ATR")
    st.line_chart(df_ind.set_index("time")["ATR"])

if show_debug:
    with st.expander("Debug raw dataframe"):
        st.dataframe(df_ind.tail(200), width='stretch')
    with st.expander("Columns list"):
        st.write(df_ind.columns.tolist())
