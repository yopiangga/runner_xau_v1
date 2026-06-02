"""Backtest REALISTIS untuk trading MANUAL (sinyal -> dieksekusi manusia).

Yang dibuat realistis (beda dari backtest ideal sebelumnya):
1. ENTRY di OPEN candle BERIKUTNYA (manusia bereaksi setelah candle tutup),
   bukan di harga close candle sinyal.
2. SPREAD (selisih bid/ask) dibebankan saat masuk.
3. SLIPPAGE saat entry & saat kena Stop Loss (stop order tergelincir).
4. SATU POSISI dalam satu waktu (manusia tak bisa pegang puluhan posisi
   yang tumpang tindih) -> sinyal saat sedang ada posisi diabaikan.
5. COOLDOWN setelah posisi tutup (jeda agar tidak overtrading).

Output: winrate & PnL BERSIH setelah biaya, jumlah trade, drawdown.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import walk_forward

FEATS = FEATURE_COLS + ["side"]

# ---- Asumsi biaya (sesuaikan dengan broker Anda) ----
SPREAD = 0.30        # spread emas (USD) -> dibayar saat entry
SLIP_ENTRY = 0.10    # slippage saat entry (USD)
SLIP_STOP = 0.15     # slippage tambahan saat kena SL (stop order)
COOLDOWN_BARS = 3    # jeda 3 candle (15 menit) setelah trade tutup


def build_signals(threshold):
    df = fetch_klines()
    df = build_features(df)
    n = len(df)

    dfl = make_labels(df, "long"); dfl["idx"] = np.arange(n)
    dfs = make_labels(df, "short"); dfs["idx"] = np.arange(n)
    for d, side in [(dfl, 0.0), (dfs, 1.0)]:
        # sl_price absolut (level support/resistance + buffer)
        d["sl_price"] = np.where(d["side"] == 0.0, d["close"] - d["sl_dist"],
                                 d["close"] + d["sl_dist"])
    keep = ["datetime", "idx", "close", "sl_price"] + FEATURE_COLS + \
           ["label", "sl_dist", "tp_dist", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"] + FEATS)
    both = both.sort_values(["datetime", "side"]).reset_index(drop=True)

    res, auc = walk_forward(both, verbose=False, feature_cols=FEATS, threshold=threshold)
    # sinyal hanya di periode OOS (yang punya proba)
    sig_long, sig_short, oos_idx = {}, {}, set()
    for _, r in res.iterrows():
        i = int(r["idx"]); oos_idx.add(i)
        if r["proba"] >= threshold:
            if r["side"] == 0.0:
                sig_long[i] = (r["proba"], r["sl_price"])
            else:
                sig_short[i] = (r["proba"], r["sl_price"])
    return df, sig_long, sig_short, oos_idx, auc


def simulate(df, sig_long, sig_short, oos_idx, threshold):
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    n = len(df)
    tp_pts = config.TP_POINTS
    horizon = config.HOLD_HORIZON
    trades = []

    i = min(oos_idx)
    last_i = max(oos_idx)
    while i <= last_i:
        # pilih sinyal terkuat di bar i (boleh BUY atau SELL)
        cand = []
        if i in sig_long:  cand.append(("BUY", *sig_long[i]))
        if i in sig_short: cand.append(("SELL", *sig_short[i]))
        if not cand or i + 1 >= n:
            i += 1; continue
        cand.sort(key=lambda x: x[1], reverse=True)   # proba tertinggi
        side, proba, sl_price = cand[0]
        dirn = 1 if side == "BUY" else -1

        # ENTRY di open candle berikutnya + spread + slippage (harga lebih buruk)
        entry = o[i + 1] + dirn * (SPREAD / 2 + SLIP_ENTRY)
        tp_price = entry + dirn * tp_pts

        exit_price, exit_idx, result = None, None, None
        end = min(n, (i + 1) + horizon)
        for j in range(i + 1, end):
            if dirn == 1:
                hit_sl = l[j] <= sl_price
                hit_tp = h[j] >= tp_price
            else:
                hit_sl = h[j] >= sl_price
                hit_tp = l[j] <= tp_price
            if hit_sl and hit_tp:        # ambigu -> SL dulu (konservatif)
                exit_price = sl_price - dirn * SLIP_STOP; result = "SL"; exit_idx = j; break
            if hit_sl:
                exit_price = sl_price - dirn * SLIP_STOP; result = "SL"; exit_idx = j; break
            if hit_tp:
                exit_price = tp_price; result = "TP"; exit_idx = j; break
        if exit_price is None:           # horizon habis -> tutup di close
            exit_idx = end - 1; exit_price = c[exit_idx]; result = "TIME"

        pnl = dirn * (exit_price - entry)
        bars_held = exit_idx - (i + 1) + 1
        trades.append({
            "waktu_sinyal": df["datetime"].iloc[i],
            "waktu_entry": df["datetime"].iloc[i + 1],
            "side": side, "proba": round(proba, 3),
            "entry": round(entry, 2), "tp": round(tp_price, 2),
            "sl": round(sl_price, 2), "exit": round(exit_price, 2),
            "hasil": result, "pnl": round(pnl, 2),
            "bar_held": bars_held,
            "menit_held": bars_held * 5,
        })
        i = exit_idx + 1 + COOLDOWN_BARS   # satu posisi + cooldown
    return pd.DataFrame(trades)


def report(trades, auc, threshold):
    n = len(trades)
    if n == 0:
        print("Tidak ada trade."); return
    tp = (trades["hasil"] == "TP").sum()
    sl = (trades["hasil"] == "SL").sum()
    tm = (trades["hasil"] == "TIME").sum()
    net_win = (trades["pnl"] > 0).sum()
    days = (trades["waktu_entry"].max() - trades["waktu_entry"].min()).days or 1
    eq = trades["pnl"].cumsum()
    dd = (eq.cummax() - eq).max()

    print("=" * 72)
    print(f"BACKTEST REALISTIS (manual) — {config.SYMBOL} 5m | threshold {threshold}")
    print(f"Biaya: spread {SPREAD} + slip entry {SLIP_ENTRY} + slip stop {SLIP_STOP} | "
          f"1 posisi/waktu | cooldown {COOLDOWN_BARS} bar")
    print("=" * 72)
    print(f"AUC model               : {auc:.3f}")
    print(f"Jumlah trade            : {n}  (~{n/days:.1f}/hari, {days} hari)")
    print(f"  - BUY / SELL          : {(trades['side']=='BUY').sum()} / {(trades['side']=='SELL').sum()}")
    print(f"Kena TP                 : {tp} ({tp/n*100:.1f}%)")
    print(f"Kena SL                 : {sl} ({sl/n*100:.1f}%)")
    print(f"Habis waktu (TIME)      : {tm}")
    print(f">> WINRATE (TP)         : {tp/n*100:.1f}%")
    print(f">> WINRATE bersih(PnL>0): {net_win/n*100:.1f}%")
    print(f"Total PnL bersih        : {trades['pnl'].sum():.2f} USD")
    print(f"Rata-rata per trade     : {trades['pnl'].mean():.2f} USD")
    print(f"Max drawdown            : {dd:.2f} USD")
    print(f"Durasi rata-rata posisi : {trades['menit_held'].mean():.0f} menit")
    trades.to_csv("realistic_trades.csv", index=False)
    print("\n[save] realistic_trades.csv")
    print("\n--- 12 trade pertama ---")
    print(trades.head(12).to_string(index=False))


def main(threshold=None):
    threshold = threshold or config.PROB_THRESHOLD
    df, sl_, ss_, oos, auc = build_signals(threshold)
    trades = simulate(df, sl_, ss_, oos, threshold)
    report(trades, auc, threshold)
    return trades


if __name__ == "__main__":
    import sys
    th = float(sys.argv[1]) if len(sys.argv) > 1 else None
    main(th)
