"""Generator SINYAL untuk trader MANUAL.

Dijalankan setelah candle 5m tutup. Mengambil data terbaru dari AllTick,
menghitung indikator pada candle terakhir yang SUDAH tutup, lalu
mengeluarkan rekomendasi: BUY / SELL / TUNGGU + Entry, TP, SL, RR, keyakinan.

Cara pakai:
  python3 live_signal.py            # sekali jalan, pakai cache + train cepat
  python3 live_signal.py --fetch    # tarik data terbaru dulu (disarankan live)
"""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import fit_full_model

FEATS = FEATURE_COLS + ["side"]


def latest_signal(force_fetch=False):
    df = fetch_klines(force=force_fetch)
    df = build_features(df)

    # latih model 2-arah pada seluruh data berlabel
    dfl = make_labels(df, "long"); dfs = make_labels(df, "short")
    keep = FEATURE_COLS + ["label", "sl_dist", "tp_dist", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], np.nan)
    model = fit_full_model(both, feature_cols=FEATS)

    # candle terakhir yang SUDAH tutup
    last = df.iloc[-1]
    atr = last["atr"]; close = last["close"]
    buf = config.SL_BUFFER_ATR * atr

    options = []
    # --- evaluasi BUY ---
    if pd.notna(last["support"]):
        sl_price = last["support"] - buf
        sl_d = close - sl_price
        if config.MIN_SL_POINTS <= sl_d <= config.MAX_SL_POINTS:
            row = last[FEATURE_COLS].copy(); row["side"] = 0.0
            p = model.predict_proba(row.to_frame().T[FEATS])[0, 1]
            options.append(("BUY", p, close + config.TP_POINTS, sl_price, sl_d))
    # --- evaluasi SELL ---
    if pd.notna(last["resistance"]):
        sl_price = last["resistance"] + buf
        sl_d = sl_price - close
        if config.MIN_SL_POINTS <= sl_d <= config.MAX_SL_POINTS:
            row = last[FEATURE_COLS].copy(); row["side"] = 1.0
            p = model.predict_proba(row.to_frame().T[FEATS])[0, 1]
            options.append(("SELL", p, close - config.TP_POINTS, sl_price, sl_d))

    return df, last, options


def main():
    df, last, options = latest_signal("--fetch" in sys.argv)
    th = config.PROB_THRESHOLD

    print("=" * 60)
    print(f"  SINYAL {config.SYMBOL} (XAUUSD) — {config.TIMEFRAME_LABEL}")
    print("=" * 60)
    print(f"Candle terakhir tutup : {last['datetime']}  (waktu UTC)")
    print(f"Harga close           : {last['close']:.2f}")
    print(f"Support terdekat       : {last['support']:.2f}" if pd.notna(last['support']) else "Support: -")
    print(f"Resistance terdekat    : {last['resistance']:.2f}" if pd.notna(last['resistance']) else "Resistance: -")
    print(f"Threshold keyakinan    : {th:.0%}")
    print("-" * 60)

    if not options:
        print(">> AKSI: TUNGGU (tidak ada setup valid: support/resistance terlalu jauh)")
        return
    options.sort(key=lambda x: x[1], reverse=True)
    side, p, tp, sl, sl_d = options[0]
    rr = config.TP_POINTS / sl_d

    print("Kandidat sinyal:")
    for s, pp, *_ in options:
        print(f"   {s:4s}  keyakinan menang = {pp*100:.1f}%")
    print("-" * 60)

    if p >= th:
        print(f">> AKSI    : {side}  ✅")
        print(f"   Entry   : ~{last['close']:.2f}  (di OPEN candle berikutnya)")
        print(f"   TP      : {tp:.2f}   (tetap {config.TP_POINTS} USD)")
        print(f"   SL      : {sl:.2f}   (di {'bawah support' if side=='BUY' else 'atas resistance'}, {sl_d:.2f} USD)")
        print(f"   RR      : {rr:.2f}")
        print(f"   Keyakinan menang: {p*100:.1f}%")
    else:
        print(f">> AKSI    : TUNGGU (keyakinan terbaik {p*100:.1f}% < threshold {th*100:.0f}%)")
        print(f"   (sinyal terkuat saat ini: {side} {p*100:.1f}%)")


if __name__ == "__main__":
    main()
