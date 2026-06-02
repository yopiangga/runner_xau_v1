"""Feature engineering: indikator teknikal + deteksi support (swing low).
Hanya memakai pandas/numpy (tanpa lib TA eksternal)."""
import numpy as np
import pandas as pd
import config


# ---------- Indikator dasar ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal

def stoch(df, n=14, d=3):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    return k.fillna(50), k.rolling(d).mean().fillna(50)

def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/n, adjust=False).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean().fillna(0)


# ---------- Support / Resistance ----------
def swing_lows(df, window=None):
    """Tandai swing low: low[i] adalah minimum lokal dalam +-window bar."""
    window = window or config.SWING_WINDOW
    low = df["low"].values
    n = len(low)
    is_low = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = low[i - window:i + window + 1]
        if low[i] == seg.min():
            is_low[i] = True
    return pd.Series(is_low, index=df.index)


def swing_highs(df, window=None):
    """Tandai swing high: high[i] adalah maksimum lokal dalam +-window bar."""
    window = window or config.SWING_WINDOW
    high = df["high"].values
    n = len(high)
    is_high = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = high[i - window:i + window + 1]
        if high[i] == seg.max():
            is_high[i] = True
    return pd.Series(is_high, index=df.index)


def nearest_support(df, lookback=None, window=None):
    """Untuk tiap bar, cari swing low TERTINGGI yang masih < close saat ini
    dalam `lookback` bar terakhir. Inilah level SL variatif untuk BUY."""
    lookback = lookback or config.SUPPORT_LOOKBACK
    sl_mask = swing_lows(df, window).values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    support = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - lookback)
        cands = [low[j] for j in range(lo, i + 1) if sl_mask[j] and low[j] < close[i]]
        if cands:
            support[i] = max(cands)  # support terdekat di bawah harga
    return pd.Series(support, index=df.index)


def nearest_resistance(df, lookback=None, window=None):
    """Untuk tiap bar, cari swing high TERENDAH yang masih > close saat ini
    dalam `lookback` bar terakhir. Inilah level SL variatif untuk SELL."""
    lookback = lookback or config.SUPPORT_LOOKBACK
    sh_mask = swing_highs(df, window).values
    high = df["high"].values
    close = df["close"].values
    n = len(df)
    resistance = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - lookback)
        cands = [high[j] for j in range(lo, i + 1) if sh_mask[j] and high[j] > close[i]]
        if cands:
            resistance[i] = min(cands)  # resistance terdekat di atas harga
    return pd.Series(resistance, index=df.index)


# ---------- Pipeline fitur ----------
def build_features(df):
    df = df.copy().reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]

    df["ret1"] = c.pct_change()
    df["ret3"] = c.pct_change(3)
    df["ret6"] = c.pct_change(6)

    for n in (9, 21, 50, 200):
        df[f"ema{n}"] = ema(c, n)
        df[f"dist_ema{n}"] = (c - df[f"ema{n}"]) / df[f"ema{n}"]
    df["ema9_21"] = (df["ema9"] - df["ema21"]) / df["ema21"]
    df["ema21_50"] = (df["ema21"] - df["ema50"]) / df["ema50"]

    df["rsi"] = rsi(c, 14)
    df["atr"] = atr(df, 14)
    df["atr_pct"] = df["atr"] / c
    m, s, hh = macd(c)
    df["macd"], df["macd_sig"], df["macd_hist"] = m, s, hh
    df["stoch_k"], df["stoch_d"] = stoch(df)
    df["adx"] = adx(df)

    # Bollinger
    ma20 = c.rolling(20).mean(); sd20 = c.rolling(20).std()
    df["bb_pos"] = (c - ma20) / (2 * sd20).replace(0, np.nan)
    df["bb_width"] = (4 * sd20) / ma20

    # posisi relatif terhadap high/low terkini
    df["hi20"] = h.rolling(20).max(); df["lo20"] = l.rolling(20).min()
    df["pos20"] = (c - df["lo20"]) / (df["hi20"] - df["lo20"]).replace(0, np.nan)

    # struktur candle
    rng = (h - l).replace(0, np.nan)
    df["body"] = (c - df["open"]) / rng
    df["upper_wick"] = (h - c.combine(df["open"], max)) / rng
    df["lower_wick"] = (c.combine(df["open"], min) - l) / rng

    # waktu
    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek

    # support/resistance & jarak ke level (dalam satuan ATR)
    df["support"] = nearest_support(df)
    df["resistance"] = nearest_resistance(df)
    df["dist_support_atr"] = (c - df["support"]) / df["atr"]
    df["dist_resistance_atr"] = (df["resistance"] - c) / df["atr"]

    return df


FEATURE_COLS = [
    "ret1", "ret3", "ret6",
    "dist_ema9", "dist_ema21", "dist_ema50", "dist_ema200",
    "ema9_21", "ema21_50",
    "rsi", "atr_pct", "macd", "macd_sig", "macd_hist",
    "stoch_k", "stoch_d", "adx", "bb_pos", "bb_width", "pos20",
    "body", "upper_wick", "lower_wick", "hour", "dow",
    "dist_support_atr", "dist_resistance_atr",
]
