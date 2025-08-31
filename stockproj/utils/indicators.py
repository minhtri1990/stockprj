import pandas as pd

def compute_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close'):
    if df.empty or price_col not in df.columns:
        return df
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    df[f'RSI_{period}'] = rsi
    return df

def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9, price_col='close'):
    if df.empty or price_col not in df.columns:
        return df
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    df['MACD'] = macd_line
    df['MACD_SIGNAL'] = signal_line
    df['MACD_HIST'] = hist
    return df

def add_ma(df, periods, price_col='close'):
    for p in periods:
        if p > 0 and price_col in df.columns:
            df[f'MA{p}'] = df[price_col].rolling(p).mean()
    return df
