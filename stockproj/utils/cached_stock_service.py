import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from .stock_data_service import StockDataService

class CachedStockService:
    """
    Wrapper cho StockDataService, lưu trữ dữ liệu OHLC và volume của cổ phiếu
    trong 3000 ngày gần nhất để tái sử dụng cho những lần sau.
    """
    
    def __init__(self, symbol: str, source: str = 'VCI', db_path: str = 'stock_data.db'):
        """
        Khởi tạo CachedStockService
        
        Args:
            symbol: Mã cổ phiếu
            source: Nguồn dữ liệu (mặc định: 'VCI')
            db_path: Đường dẫn đến file database SQLite
        """
        self.symbol = symbol.upper()
        self.source = source
        self.db_path = db_path
        self.stock_service = StockDataService(symbol=symbol, source=source)
        self._init_db()
        
    def _init_db(self):
        """Khởi tạo database nếu chưa tồn tại"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tạo bảng lưu dữ liệu giá
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            time TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            last_updated TEXT,
            PRIMARY KEY (symbol, time)
        )
        ''')
        
        # Tạo bảng lưu thông tin metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_metadata (
            symbol TEXT PRIMARY KEY,
            source TEXT,
            last_full_update TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _is_data_fresh(self, max_days_old=1):
        """
        Kiểm tra xem dữ liệu có được cập nhật gần đây không
        
        Args:
            max_days_old: Số ngày tối đa mà dữ liệu được coi là còn mới
            
        Returns:
            bool: True nếu dữ liệu còn mới, False nếu cần cập nhật
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT last_full_update FROM stock_metadata WHERE symbol = ?", 
            (self.symbol,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return False
            
        last_update = datetime.strptime(result[0], "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        
        return (now - last_update).days < max_days_old
    
    def _clear_symbol_data(self):
        """Xóa toàn bộ dữ liệu của symbol hiện tại"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM stock_prices WHERE symbol = ?", (self.symbol,))
        cursor.execute("DELETE FROM stock_metadata WHERE symbol = ?", (self.symbol,))
        
        conn.commit()
        conn.close()
    
    def _store_price_data(self, df):
        """
        Lưu dữ liệu giá vào database
        
        Args:
            df: DataFrame chứa dữ liệu giá
        """
        if df is None or df.empty:
            print(f"[WARN] Không có dữ liệu để lưu cho {self.symbol}")
            return
            
        conn = sqlite3.connect(self.db_path)
        
        # Chuẩn bị dữ liệu để lưu
        df_to_save = df.copy()
        if 'time' in df_to_save.columns:
            # Chuyển đổi cột time sang dạng string nếu là datetime
            if pd.api.types.is_datetime64_any_dtype(df_to_save['time']):
                df_to_save['time'] = df_to_save['time'].dt.strftime('%Y-%m-%d')
        
        # Thêm thông tin symbol và thời gian cập nhật
        df_to_save['symbol'] = self.symbol
        df_to_save['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Lưu vào database
        required_columns = ['symbol', 'time', 'open', 'high', 'low', 'close', 'volume', 'last_updated']
        available_columns = [col for col in required_columns if col in df_to_save.columns]
        
        if len(available_columns) < 6:  # Thiếu quá nhiều cột
            print(f"[ERROR] Dữ liệu thiếu nhiều cột quan trọng: {set(required_columns) - set(available_columns)}")
            conn.close()
            return
            
        # Điền giá trị null cho các cột thiếu
        for col in set(required_columns) - set(available_columns):
            df_to_save[col] = None
            
        # Lưu dữ liệu
        df_to_save[required_columns].to_sql('stock_prices', conn, if_exists='append', index=False)
        
        # Cập nhật metadata
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO stock_metadata (symbol, source, last_full_update) VALUES (?, ?, ?)",
            (self.symbol, self.source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        conn.commit()
        conn.close()
        print(f"[INFO] Đã lưu {len(df_to_save)} bản ghi cho {self.symbol}")
    
    def refresh_data(self, force=False):
        """
        Làm mới dữ liệu từ API nếu cần
        
        Args:
            force: Buộc làm mới dữ liệu ngay cả khi dữ liệu còn mới
            
        Returns:
            bool: True nếu dữ liệu đã được làm mới, False nếu không
        """
        if not force and self._is_data_fresh():
            print(f"[INFO] Dữ liệu cho {self.symbol} vẫn còn mới, không cần cập nhật")
            return False
            
        print(f"[INFO] Đang làm mới dữ liệu cho {self.symbol}...")
        
        # Tính toán khoảng thời gian 3000 ngày
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=3000)).strftime("%Y-%m-%d")
        
        # Lấy dữ liệu từ StockDataService
        df = self.stock_service.get_price_history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            print(f"[WARN] Không lấy được dữ liệu cho {self.symbol}")
            return False
            
        # Xóa dữ liệu cũ và lưu dữ liệu mới
        self._clear_symbol_data()
        self._store_price_data(df)
        
        return True
    
    def get_price_history(self, start=None, end=None, interval='1D', add_ma=None):
        """
        Lấy dữ liệu giá từ cache hoặc API nếu cần
        
        Args:
            start: Ngày bắt đầu (format: 'YYYY-MM-DD')
            end: Ngày kết thúc (format: 'YYYY-MM-DD')
            interval: Khoảng thời gian ('1D', '1W', '1M')
            add_ma: Danh sách chu kỳ MA cần thêm vào
            
        Returns:
            DataFrame: Dữ liệu giá
        """
        # Làm mới dữ liệu nếu cần
        self.refresh_data()
        
        # Xác định khoảng thời gian nếu không được cung cấp
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        # Truy vấn dữ liệu từ database
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT time, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = ? AND time BETWEEN ? AND ?
        ORDER BY time
        """
        
        df = pd.read_sql_query(query, conn, params=(self.symbol, start, end))
        conn.close()
        
        if df.empty:
            print(f"[WARN] Không có dữ liệu trong cache cho {self.symbol} từ {start} đến {end}")
            # Thử lấy từ API
            return self.stock_service.get_price_history(start=start, end=end, interval=interval, add_ma=add_ma)
        
        # Chuyển đổi cột time thành datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Xử lý interval
        if interval in ['1W', '1M']:
            rule = 'W' if interval == '1W' else 'M'
            agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = (df.set_index('time')
                   .resample(rule)
                   .agg(agg)
                   .dropna()
                   .reset_index())
        
        # Thêm MA nếu cần
        if add_ma and 'close' in df.columns:
            from .indicators import add_ma as add_ma_func
            df = add_ma_func(df, add_ma)
            
        return df
    
    # Các phương thức khác của StockDataService
    def get_profile(self):
        """Lấy thông tin profile của cổ phiếu"""
        return self.stock_service.get_profile()
        
    def get_ratios(self, period='quarterly', limit=8):
        """Lấy thông tin tỷ số tài chính"""
        return self.stock_service.get_ratios(period=period, limit=limit)
        
    def get_income(self, period='quarterly', consolidated=True):
        """Lấy báo cáo kết quả kinh doanh"""
        return self.stock_service.get_income(period=period, consolidated=consolidated)
        
    def get_balance(self, period='quarterly', consolidated=True):
        """Lấy bảng cân đối kế toán"""
        return self.stock_service.get_balance(period=period, consolidated=consolidated)
        
    def get_cashflow(self, period='quarterly', consolidated=True):
        """Lấy báo cáo lưu chuyển tiền tệ"""
        return self.stock_service.get_cashflow(period=period, consolidated=consolidated)