import streamlit as st
from datetime import date, timedelta
from utils.cached_stock_service import CachedStockService
from utils.charts import multi_close_chart
from utils.load_all_symbols import load_all_symbols

st.title("📈 So sánh nhiều mã")

all_symbols = load_all_symbols()
selected = st.multiselect("Chọn mã", all_symbols, default=all_symbols[:4])
end = date.today()
start = st.date_input("Từ ngày", end - timedelta(days=180))
interval = st.selectbox("Chu kỳ", ["1D","1W","1M"])

price_dict = {}
for s in selected:
    srv = CachedStockService(s)
    df = srv.get_price_history(start=start.isoformat(), end=end.isoformat(), interval=interval)
    price_dict[s] = df

fig = multi_close_chart(price_dict)
st.plotly_chart(fig, width='stretch')

if not selected:
    st.info("Chọn ít nhất một mã.")
