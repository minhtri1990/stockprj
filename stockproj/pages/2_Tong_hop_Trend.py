import streamlit as st
from datetime import date, timedelta, datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from utils.load_all_symbols import load_all_symbols
from utils.trend_detect import TrendDetector
from utils.cached_stock_service import CachedStockService

st.set_page_config(page_title="Tổng hợp Trend", page_icon="🧮", layout="wide")

st.title("🧮 Tổng hợp Trend nhiều cổ phiếu")
st.caption("Bảng thống kê nhanh: Trend hiện tại, Score, Δ Score, Chuỗi hiện tại (streak).")

# ================================
# Cấu hình chung & cache
# ================================
@st.cache_data(ttl=600, show_spinner=False)
def get_all_symbols():
    return load_all_symbols()  # (symbols, note)

@st.cache_data(ttl=120, show_spinner=False)
def fetch_price_history(symbol: str, start: str, end: str, interval: str):
    """
    Lấy dữ liệu giá (không thêm MA để nhanh hơn).
    """
    service = CachedStockService(symbol)
    df = service.get_price_history(
        start=start,
        end=end,
        interval=interval,
        add_ma=None  # TrendDetector sẽ tự tính những gì nó cần
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"])
        except Exception:
            pass
    return df

@st.cache_data(ttl=120, show_spinner=False)
def compute_trend_for_symbol(symbol: str,
                             start: str,
                             end: str,
                             interval: str,
                             method: str,
                             ema_fast: int,
                             ema_slow: int,
                             ema_slope_lookback: int,
                             adx_period: int,
                             adx_threshold: int,
                             donchian_window: int,
                             donchian_proximity: float,
                             score_up_threshold: int,
                             score_down_threshold: int,
                             use_obv: bool,
                             obv_lookback: int,
                             keep_components: bool,
                             compute_streak: bool) -> Dict[str, Any]:
    """
    Trả về dict thống kê cho 1 symbol.
    """
    df_price = fetch_price_history(symbol, start, end, interval)
    if df_price.empty:
        return {
            "symbol": symbol,
            "trend": "N/A",
            "score": np.nan,
            "delta_score": np.nan,
            "streak": None,
            "ok": False,
            "error": "No data"
        }
    try:
        detector = TrendDetector(
            method=method,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            ema_slope_lookback=ema_slope_lookback,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            donchian_window=donchian_window,
            donchian_proximity=donchian_proximity,
            score_up_threshold=score_up_threshold,
            score_down_threshold=score_down_threshold,
            use_obv=(use_obv and method == "multi"),
            obv_lookback=obv_lookback,
            keep_components=keep_components,
            compute_streak=compute_streak
        )
        df_trend = detector.detect(df_price)
        stats = detector.get_stats()
        score_col = stats.get("score_col")
        label_col = stats.get("label_col")
        streak_col = stats.get("streak_col")

        if df_trend.empty:
            return {
                "symbol": symbol,
                "trend": "N/A",
                "score": np.nan,
                "delta_score": np.nan,
                "streak": None,
                "ok": False,
                "error": "Empty trend"
            }

        last_row = df_trend.iloc[-1]
        score = last_row[score_col]
        trend_label = last_row[label_col]
        if len(df_trend) > 1:
            prev_score = df_trend.iloc[-2][score_col]
            delta_score = score - prev_score
        else:
            delta_score = np.nan

        streak_val = None
        if compute_streak and streak_col in df_trend.columns:
            streak_val = int(last_row[streak_col]) if pd.notna(last_row[streak_col]) else None

        return {
            "symbol": symbol,
            "trend": trend_label,
            "score": score,
            "delta_score": delta_score,
            "streak": streak_val,
            "ok": True,
            "error": None
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "trend": "ERR",
            "score": np.nan,
            "delta_score": np.nan,
            "streak": None,
            "ok": False,
            "error": str(e)
        }

# ================================
# Sidebar controls
# ================================
with st.sidebar:
    st.header("Thiết lập")
    symbols = get_all_symbols()

    # Chia symbols thành các phần, mỗi phần tối đa 30 mã
    length_path = 40
    parts = [symbols[i:i + length_path] for i in range(0, len(symbols), length_path)]
    part_labels = [f"Part {i + 1}" for i in range(len(parts))]
    selected_part = st.selectbox("Chọn phần", part_labels, index=0)
    selected_symbols = parts[part_labels.index(selected_part)]

    # Hiển thị danh sách mã trong phần đã chọn
    selected_symbols = st.multiselect(
        "Danh sach mã",
        selected_symbols,
        default=selected_symbols
    )

    if not selected_symbols:
        st.warning("Chọn ít nhất 1 mã.")
        st.stop()

    today = date.today()
    start_default = today - timedelta(days=365)  # 1 năm
    start_date = st.date_input("Từ ngày", start_default)
    end_date = st.date_input("Đến ngày", today)
    interval = st.selectbox("Interval", ["1D", "1W", "1M"], index=0)

    st.subheader("Trend Params")
    method = st.selectbox("Phương pháp",
                          ["multi", "ema", "adx", "donchian"],
                          index=0)
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
    compute_streak = st.checkbox("Tính streak", value=True)
    keep_components = st.checkbox("Giữ cột chi tiết (ít cần cho summary)", value=False)

    st.divider()
    run_btn = st.button("🚀 Chạy thống kê", width='stretch')

# ================================
# Validate date
# ================================
if start_date > end_date:
    st.error("Ngày bắt đầu phải nhỏ hơn hoặc bằng ngày kết thúc.")
    st.stop()

if not run_btn:
    st.info("Nhấn 'Chạy thống kê' để bắt đầu.")
    st.stop()

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

# ================================
# Execute
# ================================
st.subheader("Kết quả tổng hợp")
progress = st.progress(0)
results: List[Dict[str, Any]] = []
total = len(selected_symbols)

for i, sym in enumerate(selected_symbols, start=1):
    res = compute_trend_for_symbol(
        symbol=sym,
        start=start_str,
        end=end_str,
        interval=interval,
        method=method,
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_slope_lookback=ema_slope_lookback,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        donchian_window=donchian_window,
        donchian_proximity=donchian_proximity,
        score_up_threshold=score_up_threshold,
        score_down_threshold=score_down_threshold,
        use_obv=use_obv,
        obv_lookback=obv_lookback,
        keep_components=keep_components,
        compute_streak=compute_streak
    )
    results.append(res)
    progress.progress(i / total)

if not results:
    st.warning("Không có kết quả.")
    st.stop()

df_summary = pd.DataFrame(results)

# Lọc bỏ lỗi nếu muốn
show_errors = st.checkbox("Hiển thị cả dòng lỗi", value=False)
if not show_errors:
    df_display = df_summary[df_summary["ok"]]
else:
    df_display = df_summary.copy()

# Chuẩn hoá cột hiển thị
df_display = df_display.assign(
    Score=lambda d: d["score"],
    Delta_Score=lambda d: d["delta_score"],
    Trend=lambda d: d["trend"],
    Chuỗi_hiện_tại=lambda d: d["streak"]
)

columns_show = ["symbol", "Trend", "Score", "Delta_Score"]
if compute_streak:
    columns_show.append("Chuỗi_hiện_tại")
if show_errors:
    columns_show.append("error")

# Sắp xếp: ưu tiên score giảm dần
df_display = df_display[columns_show].sort_values("Score", ascending=False, na_position="last")

# Format
def fmt_delta(x):
    return f"{x:+.0f}" if pd.notna(x) else ""
def fmt_score(x):
    return f"{x:.0f}" if pd.notna(x) else ""

df_format = df_display.copy()
if "Score" in df_format.columns:
    df_format["Score"] = df_format["Score"].apply(fmt_score)
if "Delta_Score" in df_format.columns:
    df_format["Delta_Score"] = df_format["Delta_Score"].apply(fmt_delta)

st.dataframe(df_format, width='stretch', height=min(600, 50 + 30 * len(df_format)))

# Download CSV (dạng raw không format để tiện xử lý tiếp)
csv_bytes = df_display.to_csv(index=False).encode("utf-8")
st.download_button("📥 Tải CSV", data=csv_bytes, file_name="trend_summary.csv", mime="text/csv")

# ================================
# Thống kê phụ
# ================================
with st.expander("Thống kê thêm"):
    if compute_streak and "Chuỗi_hiện_tại" in df_display.columns:
        st.write("Phân bố chuỗi hiện tại (streak):")
        streak_counts = (
            df_display["Chuỗi_hiện_tại"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
        )
        st.bar_chart(streak_counts)
    trend_counts = df_display["Trend"].value_counts(dropna=False)
    st.write("Phân bố Trend:")
    st.write(trend_counts)

# ================================
# Ghi chú
# ================================
st.info(
    "Ghi chú:\n"
    "- Δ Score = Score hiện tại trừ Score phiên trước.\n"
    "- Chuỗi hiện tại = số phiên liên tiếp có cùng nhãn Trend.\n"
    "- Các dòng lỗi (ERR / N/A) có thể do thiếu dữ liệu hoặc tham số không phù hợp."
)