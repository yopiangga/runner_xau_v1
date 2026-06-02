"""Pelabelan triple-barrier untuk aksi BUY (long-only).

Aturan sesuai permintaan:
  - TP TETAP  : entry + TP_POINTS
  - SL VARIATIF: tepat di bawah support terdekat (variatif tiap sinyal)
Label = 1 bila TP tersentuh lebih dulu, 0 bila SL lebih dulu, NaN bila
tidak ada barrier tersentuh dalam horizon (atau setup tidak valid).
"""
import numpy as np
import pandas as pd
import config


def make_labels(df, side="long"):
    """Pelabelan triple-barrier untuk satu arah.
    side='long' (BUY): TP di atas, SL di bawah support.
    side='short'(SELL): TP di bawah, SL di atas resistance."""
    df = df.copy().reset_index(drop=True)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr = df["atr"].values
    support = df["support"].values
    resistance = df["resistance"].values
    n = len(df)

    tp_pts = config.TP_POINTS
    horizon = config.HOLD_HORIZON
    buf_atr = config.SL_BUFFER_ATR
    max_sl = config.MAX_SL_POINTS
    min_sl = config.MIN_SL_POINTS

    label = np.full(n, np.nan)
    sl_dist = np.full(n, np.nan)
    tp_dist = np.full(n, np.nan)
    rr = np.full(n, np.nan)

    for i in range(n - 1):
        entry = close[i]
        if np.isnan(atr[i]):
            continue

        if side == "long":
            level = support[i]
            if np.isnan(level):
                continue
            sl_price = level - buf_atr * atr[i]     # SL di bawah support
            sl_d = entry - sl_price
            tp_price = entry + tp_pts                # TP di atas
        else:  # short
            level = resistance[i]
            if np.isnan(level):
                continue
            sl_price = level + buf_atr * atr[i]     # SL di atas resistance
            sl_d = sl_price - entry
            tp_price = entry - tp_pts                # TP di bawah

        if sl_d < min_sl or sl_d > max_sl:
            continue                                 # setup di luar batas risiko

        outcome = np.nan
        end = min(n, i + 1 + horizon)
        for j in range(i + 1, end):
            if side == "long":
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price
            if hit_tp and hit_sl:
                outcome = 0.0   # ambigu dalam 1 bar -> konservatif: SL dulu
                break
            if hit_sl:
                outcome = 0.0; break
            if hit_tp:
                outcome = 1.0; break
        label[i] = outcome
        sl_dist[i] = sl_d
        tp_dist[i] = tp_pts
        rr[i] = tp_pts / sl_d

    df["label"] = label
    df["sl_dist"] = sl_dist
    df["tp_dist"] = tp_dist
    df["rr"] = rr
    df["side"] = 1.0 if side == "short" else 0.0
    return df


def summarize_labels(df):
    valid = df.dropna(subset=["label"])
    wins = (valid["label"] == 1).sum()
    losses = (valid["label"] == 0).sum()
    total = wins + losses
    print("\n=== Statistik Label (semua setup valid, tanpa model) ===")
    print(f"Setup valid     : {total}")
    if total:
        print(f"Menang (TP dulu): {wins}  ({wins/total:.1%})")
        print(f"Kalah  (SL dulu): {losses}  ({losses/total:.1%})")
        print(f"Rata-rata RR    : {valid['rr'].mean():.2f}")
        print(f"Rata-rata SL    : {valid['sl_dist'].mean():.2f} USD  | TP tetap: {config.TP_POINTS} USD")
    return total
