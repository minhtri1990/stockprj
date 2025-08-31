"""
stock_chart_builder.py
----------------------------------
Module cung cấp class StockChartBuilder để dựng biểu đồ nến (candlestick)
với Moving Averages và Volume có tô màu tăng/giảm bằng Plotly, thuận tiện
cho việc sử dụng trong ứng dụng Streamlit.

Phụ thuộc:
    - pandas
    - plotly
    - streamlit

Cài đặt nhanh:
    pip install pandas plotly streamlit

Sử dụng cơ bản trong Streamlit:
    from stock_chart_builder import StockChartBuilder

    builder = StockChartBuilder(
        df=df_price,
        symbol="ABC",
        interval="1D",
        ma_list=[20, 50],
        show_volume=True,
        volume_method="candle"
    )
    builder.build()
    builder.render()

Nếu muốn giữ API cũ:
    from stock_chart_builder import plot_chart
    plot_chart(df_price, "ABC", "1D", [20, 50], show_volume=True)
"""

from typing import List, Optional, Iterable
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def plot_chart(df, symbol, interval, ma_list=None, show_volume=True,
               volume_method="candle", template="plotly_white",
               volume_opacity=0.95, title=None, return_fig=False,
               line_properties=None):
    """
    Plots the stock chart using the StockChartBuilder.

    Parameters:
        df (pd.DataFrame): DataFrame containing price data.
        symbol (str): Stock symbol.
        interval (str): Time interval (e.g., '1D', '1W', '1M').
        ma_list (list): List of moving averages to include.
        show_volume (bool): Whether to display volume.
        volume_method (str): Method for displaying volume (e.g., 'candle').
        template (str): Plotly template to use.
        volume_opacity (float): Opacity for volume bars.
        title (str): Title of the chart.
        return_fig (bool): Whether to return the figure object.
        line_properties (dict): Dictionary of line properties (e.g., color, dash, width).

    Returns:
        None or plotly.graph_objects.Figure: If return_fig is True, returns the figure object.
    """
    try:
        builder = StockChartBuilder(
            df=df,
            symbol=symbol,
            interval=interval,
            ma_list=ma_list,
            show_volume=show_volume,
            volume_method=volume_method,
            volume_opacity=volume_opacity,
            template=template,
            title=title
        )
        if line_properties:
            # Apply line properties (e.g., color, dash, width) to the builder
            builder.set_line_properties(line_properties)

        fig = builder.build()
        if return_fig:
            return fig
        builder.render()
    except ValueError as e:
        st.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
        if return_fig:
            return None

class StockChartBuilder:
    """
    Builder dựng biểu đồ cổ phiếu (candlestick + MA + Volume).

    Tham số:
        df (pd.DataFrame): Bắt buộc gồm cột: time, open, high, low, close (+ volume nếu muốn).
        symbol (str): Mã chứng khoán.
        interval (str): Khung thời gian hiển thị (ví dụ: '1D', '1H', '15m' ...).
        ma_list (Iterable[int]): Danh sách chu kỳ MA đã được tính sẵn trong df (cột dạng MA_{n}).
        show_volume (bool): Có hiển thị volume không.
        volume_method (str): 
            - 'candle': tăng nếu close > open
            - 'prev_close': tăng nếu close > close.shift(1)
        volume_opacity (float): Độ trong suốt cột volume.
        template (str): Plotly template (plotly_white, plotly_dark, ...).
        title (str | None): Tiêu đề. Nếu None sẽ auto tạo.

    Phương thức chính:
        build()  -> go.Figure
        render() -> hiển thị trong Streamlit
        add_candlestick(), add_ma_traces(), add_volume_trace() ... (method chaining)

    Mở rộng:
        - Có thể thêm các hàm add_rsi(), add_macd() sau này.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        ma_list: Optional[Iterable[int]] = None,
        show_volume: bool = True,
        volume_method: str = "candle",   # 'candle' hoặc 'prev_close'
        volume_opacity: float = 0.95,
        template: str = "plotly_white",
        title: Optional[str] = None
    ):
        self.original_df = df
        self.df: Optional[pd.DataFrame] = None
        self.symbol = symbol
        self.interval = interval
        self.ma_list = list(ma_list) if ma_list else []
        self.show_volume = show_volume
        self.volume_method = volume_method
        self.volume_opacity = volume_opacity
        self.template = template
        self.title = title or f"{symbol} - Biểu đồ giá ({interval})"
        self.fig: Optional[go.Figure] = None
        self._validated = False

    # ------------- QUY TRÌNH CHÍNH -------------
    def build(self) -> go.Figure:
        """
        Chạy toàn bộ pipeline: validate -> prep df -> create fig -> add components -> finalize.
        """
        self._validate_input()
        self._prepare_df()
        self.create_base_figure()
        self.add_candlestick()
        self.add_ma_traces()
        if self.show_volume:
            self.add_volume_trace()
        else:
            # không có volume -> full chiều cao cho y chính
            self.fig.update_layout(yaxis=dict(domain=[0.0, 1.0]))
        self.finalize_layout()
        return self.fig

    def render(self, use_container_width: bool = True):
        """
        Hiển thị figure trong Streamlit. Tự build nếu chưa có fig.
        """
        if self.fig is None:
            self.build()
        st.plotly_chart(self.fig, width='stretch')

    # ------------- VALIDATION & PREP -------------
    def _validate_input(self):
        if self.original_df is None or self.original_df.empty:
            st.warning("Không có dữ liệu hiển thị.")
            raise ValueError("Empty dataframe")

        required = {"time", "open", "high", "low", "close"}
        missing = required - set(self.original_df.columns)
        if missing:
            st.error(f"Dữ liệu thiếu cột: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        self._validated = True

    def _prepare_df(self):
        if not self._validated:
            self._validate_input()
        df = self.original_df.copy()
        # Chuẩn hoá time & sort
        try:
            df['time'] = pd.to_datetime(df['time'])
        except Exception:
            pass
        df = df.sort_values('time').reset_index(drop=True)
        self.df = df

    # ------------- CÁC BƯỚC THÀNH PHẦN -------------
    def create_base_figure(self):
        self.fig = go.Figure()
        return self

    def add_candlestick(self, name: str = "Giá"):
        self._ensure_fig()
        df = self.df
        self.fig.add_trace(go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=name
        ))
        return self

    def add_ma_traces(self, line_width: float = 1.1):
        self._ensure_fig()
        df = self.df
        for w in self.ma_list:
            col = f"MA_{w}"
            if col in df.columns and df[col].notna().sum() > 0:
                self.fig.add_trace(go.Scatter(
                    x=df["time"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    line=dict(width=line_width)
                ))
        return self

    def _calc_volume_colors(self):
        df = self.df
        if self.volume_method == "prev_close":
            prev = df['close'].shift(1)
            up_mask = df['close'] > prev
            down_mask = df['close'] < prev
        else:  # 'candle'
            up_mask = df['close'] > df['open']
            down_mask = df['close'] < df['open']

        colors = []
        for up, down in zip(up_mask, down_mask):
            if up:
                colors.append("rgba(0,160,0,0.65)")      # xanh
            elif down:
                colors.append("rgba(200,40,40,0.65)")    # đỏ
            else:
                colors.append("rgba(120,120,120,0.45)")  # không đổi
        return colors

    def add_volume_trace(self, name: str = "Volume"):
        self._ensure_fig()
        df = self.df
        if "volume" not in df.columns:
            return self
        colors = self._calc_volume_colors()
        self.fig.add_trace(go.Bar(
            x=df["time"],
            y=df["volume"],
            name=name,
            marker_color=colors,
            yaxis="y2",
            opacity=self.volume_opacity
        ))
        # domain 2 trục
        self.fig.update_layout(
            yaxis=dict(domain=[0.25, 1.0]),
            yaxis2=dict(domain=[0.0, 0.2], anchor="x", title="Vol")
        )
        return self

    def finalize_layout(self):
        self._ensure_fig()
        self.fig.update_layout(
            title=self.title,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1
            ),
            margin=dict(l=40, r=20, t=60, b=40),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True,
                linecolor="#888"
            ),
            yaxis=dict(showline=True, linecolor="#888"),
            hovermode="x unified",
            template=self.template
        )
        return self

    # ------------- TIỆN ÍCH MỞ RỘNG -------------
    def add_custom_trace(self, trace):
        """
        Thêm trace tuỳ chỉnh (go.Scatter / go.Bar / ...).
        """
        self._ensure_fig()
        self.fig.add_trace(trace)
        return self

    # Ví dụ skeleton mở rộng (chưa cài đặt):
    # def add_rsi(self, period: int = 14, panel_height: float = 0.15):
    #     """
    #     Thêm RSI (cần subplot/phân tách domain trục y khác).
    #     Gợi ý: sử dụng make_subplots hoặc tái cấu trúc sang subplot chung.
    #     """
    #     return self

    def _ensure_fig(self):
        if self.fig is None:
            self.create_base_figure()


# ============ HÀM WRAPPER GIỮ API CŨ ============
# ----------------- VẼ BIỂU ĐỒ -----------------
def plot_chart(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    ma_list: Optional[List[int]] = None,
    show_volume: bool = True,
    volume_method: str = "candle",
    template: str = "plotly_white",
    volume_opacity: float = 0.95,
    title: Optional[str] = None,
    return_fig: bool = False
):
    try:
        builder = StockChartBuilder(
            df=df,
            symbol=symbol,
            interval=interval,
            ma_list=ma_list,
            show_volume=show_volume,
            volume_method=volume_method,
            volume_opacity=volume_opacity,
            template=template,
            title=title
        )
        fig = builder.build()
        if return_fig:
            return fig
        builder.render()
        return fig
    except ValueError:
        if return_fig:
            return None
