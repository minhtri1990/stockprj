import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def candlestick_chart(df: pd.DataFrame, title="Biểu đồ nến", ma_list=None):
    fig = go.Figure()
    if df is None or df.empty:
        fig.update_layout(title="Không có dữ liệu")
        return fig
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Giá"
    ))
    if ma_list:
        for p in ma_list:
            col = f"MA{p}"
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['time'], y=df[col],
                    mode='lines', name=col, line=dict(width=1.2)
                ))
    fig.update_layout(
        title=title,
        xaxis_title="Thời gian",
        yaxis_title="Giá",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
    return fig

def multi_close_chart(price_dict: dict):
    fig = go.Figure()
    for sym, df in price_dict.items():
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['close'], mode='lines', name=sym
        ))
    fig.update_layout(
        title="So sánh giá đóng cửa",
        template="plotly_white",
        height=550
    )
    return fig

def rsi_chart(df: pd.DataFrame, rsi_col='RSI_14'):
    fig = go.Figure()
    if df is None or df.empty or rsi_col not in df.columns:
            fig.update_layout(title="RSI")
            return fig
    fig.add_trace(go.Scatter(x=df['time'], y=df[rsi_col], mode='lines', name=rsi_col))
    fig.add_hline(y=70, line_color="red", line_dash="dash")
    fig.add_hline(y=30, line_color="green", line_dash="dash")
    fig.update_layout(title="RSI", template="plotly_white", height=250, yaxis=dict(range=[0,100]))
    return fig

def macd_chart(df: pd.DataFrame):
    needed = {'MACD','MACD_SIGNAL','MACD_HIST'}
    fig = go.Figure()
    if df is None or df.empty or not needed.issubset(df.columns):
        fig.update_layout(title="MACD")
        return fig
    fig.add_trace(go.Bar(x=df['time'], y=df['MACD_HIST'], name='Hist', marker_color='gray'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD'], name='MACD', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['MACD_SIGNAL'], name='Signal', line=dict(color='orange')))
    fig.update_layout(title="MACD", template="plotly_white", height=250)
    return fig
