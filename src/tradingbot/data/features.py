import numpy as np
import pandas as pd

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df['high'], df['low'], df['close']
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    return tr.rolling(n).mean()

def keltner_channels(df: pd.DataFrame, ema_n: int = 20, atr_n: int = 14, mult: float = 1.5):
    ema = df['close'].ewm(span=ema_n, adjust=False).mean()
    _atr = atr(df, atr_n)
    upper = ema + mult * _atr
    lower = ema - mult * _atr
    return upper, lower
