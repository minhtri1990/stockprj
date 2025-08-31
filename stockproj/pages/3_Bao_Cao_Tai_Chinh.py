import streamlit as st
import plotly.express as px
from utils.cached_stock_service import CachedStockService
from utils.load_all_symbols import load_all_symbols

st.title("🧾 Báo cáo tài chính & Chỉ số")

symbol = st.selectbox("Mã", load_all_symbols(), index=0)
period = st.selectbox("Kỳ", ["quarterly","yearly"])
limit = st.slider("Số kỳ ratios", 4, 16, 8)

service = CachedStockService(symbol)
ratios = service.get_ratios(period=period, limit=limit)
income = service.get_income(period=period)
balance = service.get_balance(period=period)
cash = service.get_cashflow(period=period)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Chỉ số (Ratios)")
    if ratios is not None and not ratios.empty:
        st.dataframe(ratios, width='stretch', height=400)
    else:
        st.info("Không có dữ liệu ratios.")
with col2:
    st.subheader("Income Statement (trích)")
    if income is not None and not income.empty:
        display_cols = [c for c in income.columns if c.lower() in ["time","revenue","net_profit","net_profit_after_tax"]]
        df_viz = income[display_cols].copy()
        if "time" in df_viz.columns:
            df_viz["time"] = df_viz["time"].astype(str)
            melt_df = df_viz.melt(id_vars="time", var_name="Chỉ tiêu", value_name="Giá trị")
            fig = px.bar(melt_df, x="time", y="Giá trị", color="Chỉ tiêu", barmode="group", title="Doanh thu & Lợi nhuận")
            st.plotly_chart(fig, width='stretch')
        st.dataframe(income, width='stretch', height=300)
    else:
        st.info("Chưa có Income Statement.")

with st.expander("Balance Sheet"):
    if balance is not None and not balance.empty:
        st.dataframe(balance, width='stretch')
    else:
        st.info("Không có Balance Sheet.")
with st.expander("Cash Flow"):
    if cash is not None and not cash.empty:
        st.dataframe(cash, width='stretch')
    else:
        st.info("Không có Cash Flow.")
