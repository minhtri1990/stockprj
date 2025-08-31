import pandas as pd
from functools import lru_cache
from vnstock import Vnstock
from .indicators import add_ma as add_ma_func

class StockDataService:
    """
    Dịch vụ lấy dữ liệu giá & báo cáo tài chính sử dụng duy nhất pattern:
        base = Vnstock()
        stock_obj = base.stock(symbol=..., source=...)
    Không dùng asset_type, không thử các phương án khác.
    """

    def __init__(self, symbol: str, source: str = 'VCI'):
        self.symbol = symbol.upper()
        self.source = source
        self._price_cache = {}

        # Luôn khởi tạo qua factory style
          # (Tạo base một lần, nếu bạn muốn tái sử dụng có thể truyền base từ ngoài)
        base = Vnstock()
        if not hasattr(base, 'stock'):
            raise RuntimeError("Phiên bản vnstock không có phương thức .stock(). Hãy kiểm tra lại thư viện.")
        self.stock = base.stock(symbol=self.symbol, source=self.source)

        # Ghi chú nhanh để debug
        self._has_quote = hasattr(self.stock, 'quote')
        self._has_finance = hasattr(self.stock, 'finance')
        print(f"[INIT] symbol={self.symbol}, source={self.source}, has_quote={self._has_quote}, has_finance={self._has_finance}")

    # -------------------- INTERNAL --------------------

    def _fetch_history_raw(self, start: str, end: str):
        """
        Lấy dữ liệu daily gốc từ quote.history nếu có.
        Bạn có thể bổ sung thêm fallback nếu API đổi tên.
        """
        if not self._has_quote or not hasattr(self.stock.quote, 'history'):
            raise RuntimeError("Không có quote.history trong đối tượng stock.")
        # Cố gắng gọi với interval='1D'; nếu lỗi, thử bỏ interval.
        try:
            return self.stock.quote.history(start=start, end=end, interval='1D')
        except TypeError:
            # Có thể phiên bản bỏ tham số interval
            return self.stock.quote.history(start=start, end=end)

    def _normalize(self, df: pd.DataFrame):
        if df is None or df.empty:
            return df
        df = df.copy()
        # Chuẩn hoá tên cột thời gian
        for c in ['time', 'date', 'datetime', 'tradingDate']:
            if c in df.columns:
                df.rename(columns={c: 'time'}, inplace=True)
                break
        # Chuẩn hoá giá
        rename_map = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
            'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close',
            'Volume': 'volume', 'vol': 'volume', 'volumeMatch': 'volume'
        }
        for src, dst in rename_map.items():
            if src in df.columns and dst not in df.columns:
                df.rename(columns={src: dst}, inplace=True)

        if 'time' in df.columns:
            try:
                df['time'] = pd.to_datetime(df['time'])
            except Exception:
                pass

        essential = ['open', 'high', 'low', 'close']
        missing = [c for c in essential if c not in df.columns]
        if missing:
            print("[WARN] Thiếu cột giá:", missing)

        return df

    # -------------------- PUBLIC --------------------

    def get_price_history(self, start: str, end: str, interval='1D', add_ma=None):
        """
        start, end: 'YYYY-MM-DD'
        interval: '1D' | '1W' | '1M'
        add_ma: list/tuple số chu kỳ (ví dụ [20, 50])
        """
        ma_key = tuple(sorted(add_ma)) if add_ma else None
        cache_key = (start, end, interval, ma_key)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key].copy()

        try:
            raw = self._fetch_history_raw(start, end)
            if raw is None or raw.empty:
                print("[INFO] raw history rỗng.")
                self._price_cache[cache_key] = raw
                return raw
            df = self._normalize(raw)
            if df is None or df.empty:
                self._price_cache[cache_key] = df
                return df

            if 'time' in df.columns:
                df = df[(df['time'] >= pd.to_datetime(start)) & (df['time'] <= pd.to_datetime(end))]
                df = df.sort_values('time')

            if interval in ['1W', '1M'] and 'time' in df.columns:
                rule = 'W' if interval == '1W' else 'M'
                agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
                if 'volume' in df.columns:
                    agg['volume'] = 'sum'
                df = (df.set_index('time')
                        .resample(rule)
                        .agg(agg)
                        .dropna()
                        .reset_index())

            if ma_key and not df.empty and 'close' in df.columns:
                try:
                    df = add_ma_func(df, ma_key)
                except Exception as e:
                    print("[WARN] add_ma lỗi:", e)

            self._price_cache[cache_key] = df.copy()
            return df
        except Exception as e:
            print("[ERROR] get_price_history:", e)
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    def get_profile(self):
        try:
            if self._has_quote:
                for name in ['company_profile', 'profile']:
                    if hasattr(self.stock.quote, name):
                        return getattr(self.stock.quote, name)()
            return {}
        except Exception as e:
            print("[ERROR] get_profile:", e)
            return {}

    @lru_cache(maxsize=32)
    def get_ratios(self, period='quarterly', limit=8):
        try:
            if self._has_finance and hasattr(self.stock.finance, 'ratios'):
                return self.stock.finance.ratios(period=period, limit=limit)
            return pd.DataFrame()
        except Exception as e:
            print("[ERROR] get_ratios:", e)
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    def get_income(self, period='quarterly', consolidated=True):
        try:
            if self._has_finance and hasattr(self.stock.finance, 'income_statement'):
                return self.stock.finance.income_statement(period=period, consolidated=consolidated)
            return pd.DataFrame()
        except Exception as e:
            print("[ERROR] get_income:", e)
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    def get_balance(self, period='quarterly', consolidated=True):
        try:
            if self._has_finance and hasattr(self.stock.finance, 'balance_sheet'):
                return self.stock.finance.balance_sheet(period=period, consolidated=consolidated)
            return pd.DataFrame()
        except Exception as e:
            print("[ERROR] get_balance:", e)
            return pd.DataFrame()

    @lru_cache(maxsize=32)
    def get_cashflow(self, period='quarterly', consolidated=True):
        try:
            if self._has_finance and hasattr(self.stock.finance, 'cash_flow'):
                return self.stock.finance.cash_flow(period=period, consolidated=consolidated)
            return pd.DataFrame()
        except Exception as e:
            print("[ERROR] get_cashflow:", e)
            return pd.DataFrame()