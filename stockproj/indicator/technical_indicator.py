import pandas as pd

class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def sma(self, window):
        return self.data['close'].rolling(window=window).mean()

    def ema(self, window):
        return self.data['close'].ewm(span=window, adjust=False).mean()

    def atr(self, data, timeframe=14):
        high_low = (data["high"] - data["low"])/data["low"]
        high_close = ((data["high"] - data["close"].shift(1))/data["close"].shift(1)).abs()
        low_close = ((data["low"] - data["close"].shift(1))/data["close"].shift(1)).abs()

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=timeframe, min_periods=timeframe).mean()
        return atr*100  # Convert to percentage

    def rsi(self, window=14):
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def macd(self, short_window=12, long_window=26, signal_window=9):
        short_ema = self.ema(short_window)
        long_ema = self.ema(long_window)
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
        return macd_line, signal_line