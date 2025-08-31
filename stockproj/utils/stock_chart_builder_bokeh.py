from typing import List, Optional, Iterable
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, DatetimeTickFormatter, Legend
from bokeh.layouts import column
import numpy as np
import streamlit as st  # Thêm Streamlit để tích hợp hiển thị trong ứng dụng

class BokehStockChartBuilder:
    """
    Builder dựng biểu đồ cổ phiếu (candlestick + MA + Volume) sử dụng thư viện Bokeh.

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
        theme (str): Bokeh theme ('light_minimal', 'dark_minimal', ...).
        template (str): Không sử dụng trong Bokeh nhưng giữ để tương thích API.
        title (str | None): Tiêu đề. Nếu None sẽ auto tạo.
        width (int): Chiều rộng của biểu đồ (pixel).
        height (int): Chiều cao của biểu đồ (pixel).
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
        theme: str = "light_minimal",
        template: str = "Bokeh",  # Không sử dụng trong Bokeh
        title: Optional[str] = None,
        width: int = 800,
        height: int = 500
    ):
        self.df = df.copy()
        self.symbol = symbol
        self.interval = interval
        self.ma_list = ma_list or []
        self.show_volume = show_volume
        self.volume_method = volume_method
        self.volume_opacity = volume_opacity
        self.theme = theme
        self.title = title or f"{symbol} - Biểu đồ giá ({interval})"
        self.width = width
        self.height = height
        self.price_chart = None
        self.volume_chart = None
        self.combined_layout = None
        self.source = None

        # Màu sắc cho biểu đồ
        self.colors = {
            'up': '#26a69a',        # xanh lá
            'down': '#ef5350',      # đỏ
            'unchanged': '#b0b0b0', # xám
            'background': '#ffffff',
            'grid': '#e0e0e0',
            'text': '#333333',
            'ma_colors': ['#2962ff', '#ff6d00', '#9c27b0', '#e91e63', '#388e3c']
        }

        # Điều chỉnh màu sắc nếu theme là dark
        if 'dark' in theme:
            self.colors.update({
                'background': '#1e1e1e',
                'grid': '#333333',
                'text': '#e0e0e0'
            })

        # Chuẩn hóa dữ liệu
        self._prepare_data()

    def _prepare_data(self):
        """Chuẩn hóa dữ liệu đầu vào."""
        # Chuyển cột 'time' thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(self.df['time']):
            self.df['time'] = pd.to_datetime(self.df['time'])

        # Tính toán màu sắc cho candlestick và volume
        self.df['color'] = np.where(self.df['close'] > self.df['open'], self.colors['up'], self.colors['down'])
        self.df['color'] = np.where(self.df['close'] == self.df['open'], self.colors['unchanged'], self.df['color'])

        if self.volume_method == "prev_close":
            prev_close = self.df['close'].shift(1)
            self.df['volume_color'] = np.where(self.df['close'] > prev_close, self.colors['up'], self.colors['down'])
            self.df['volume_color'] = np.where(self.df['close'] == prev_close, self.colors['unchanged'], self.df['volume_color'])
        else:  # 'candle'
            self.df['volume_color'] = self.df['color']

        # Tạo nguồn dữ liệu cho Bokeh
        self.source = ColumnDataSource(self.df)

    def _create_price_chart(self):
        """Tạo biểu đồ giá."""
        p = figure(
            title=self.title,
            x_axis_type="datetime",
            width=self.width,
            height=int(self.height * 0.7) if self.show_volume else self.height,
            background_fill_color=self.colors['background'],
            border_fill_color=self.colors['background'],
            outline_line_color=self.colors['grid']
        )

        # Vẽ nến
        p.segment(x0='time', y0='low', x1='time', y1='high', source=self.source, color='color', line_width=1)
        p.vbar(x='time', width=24 * 60 * 60 * 1000, top='open', bottom='close', source=self.source,
               fill_color='color', line_color='color', alpha=0.8)

        # Vẽ các đường MA
        for i, ma in enumerate(self.ma_list):
            ma_col = f"MA_{ma}"
            if ma_col in self.df.columns:
                p.line(x='time', y=ma_col, source=self.source, line_width=2,
                       color=self.colors['ma_colors'][i % len(self.colors['ma_colors'])],
                       legend_label=f"MA {ma}")

        # Thêm hover tool
        hover = HoverTool(
            tooltips=[
                ("Thời gian", "@time{%F}"),
                ("Mở", "@open{0.2f}"),
                ("Cao", "@high{0.2f}"),
                ("Thấp", "@low{0.2f}"),
                ("Đóng", "@close{0.2f}"),
                ("Khối lượng", "@volume{0,0}")
            ],
            formatters={'@time': 'datetime'},
            mode='vline'
        )
        p.add_tools(hover)

        # Định dạng trục
        p.xaxis.formatter = DatetimeTickFormatter(days="%d/%m/%Y")
        p.grid.grid_line_alpha = 0.3
        p.grid.grid_line_color = self.colors['grid']
        p.xaxis.major_label_text_color = self.colors['text']
        p.yaxis.major_label_text_color = self.colors['text']

        self.price_chart = p

    def _create_volume_chart(self):
        """Tạo biểu đồ volume."""
        p = figure(
            x_axis_type="datetime",
            x_range=self.price_chart.x_range,
            width=self.width,
            height=int(self.height * 0.3),
            background_fill_color=self.colors['background'],
            border_fill_color=self.colors['background'],
            outline_line_color=self.colors['grid']
        )

        # Vẽ cột volume
        p.vbar(x='time', top='volume', width=24 * 60 * 60 * 1000, source=self.source,
               fill_color='volume_color', line_color='volume_color', alpha=self.volume_opacity)

        # Định dạng trục
        p.xaxis.visible = False
        p.grid.grid_line_alpha = 0.3
        p.grid.grid_line_color = self.colors['grid']
        p.yaxis.major_label_text_color = self.colors['text']

        self.volume_chart = p

    def build(self):
        """Xây dựng biểu đồ."""
        self._create_price_chart()
        if self.show_volume:
            self._create_volume_chart()
            self.combined_layout = column(self.price_chart, self.volume_chart)
        else:
            self.combined_layout = self.price_chart

    def render(self):
        """Hiển thị biểu đồ."""
        if self.combined_layout is None:
            self.build()

        # Kiểm tra nếu đang chạy trong Streamlit
        try:
            import streamlit as st
            st.bokeh_chart(self.combined_layout, width='stretch')
        except ImportError:
            # Nếu không có Streamlit, sử dụng Bokeh để hiển thị
            show(self.combined_layout)