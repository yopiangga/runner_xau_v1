"""Orkestrasi eksperimen end-to-end:
  1) Ambil data XAUUSD (GOLD) dari AllTick
  2) Bangun fitur + support
  3) Pelabelan triple-barrier (TP tetap, SL variatif di support)
  4) Latih model + backtest walk-forward (out-of-sample)
  5) Laporkan winrate, threshold sweep, ekspektansi
  6) Simpan model final + prediksi sinyal terbaru
"""
import warnings
import joblib
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels, summarize_labels
from model import walk_forward, evaluate, threshold_sweep, fit_full_model

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def main(force_fetch=False):
    print("=" * 70)
    print(f"EKSPERIMEN MODEL AI — BUY {config.SYMBOL} (XAUUSD) {config.TIMEFRAME_LABEL}")
    print(f"TP tetap = {config.TP_POINTS} USD | SL variatif (support) | "
          f"horizon = {config.HOLD_HORIZON} bar")
    print("=" * 70)

    # 1) Data
    df = fetch_klines(force=force_fetch)
    print(f"Data: {len(df)} bar  ({df['datetime'].min()} → {df['datetime'].max()})")

    # 2) Fitur
    df = build_features(df)

    # 3) Label
    df = make_labels(df)
    n_setup = summarize_labels(df)
    if n_setup < 300:
        print("\n[!] Setup valid sedikit — pertimbangkan turunkan timeframe / longgarkan batas SL.")

    # 4) Backtest walk-forward
    print("\n=== Backtest Walk-Forward (out-of-sample) ===")
    res, auc = walk_forward(df)
    print(f"AUC rata-rata (kualitas model): {auc:.3f}")

    # 5) Laporan
    print("\n=== Sweep Threshold Probabilitas (out-of-sample) ===")
    sweep = threshold_sweep(res)
    show = sweep.copy()
    show["winrate"] = (show["winrate"] * 100).round(1)
    show["base_winrate_no_model"] = (show["base_winrate_no_model"] * 100).round(1)
    print(show[["threshold", "n_trades", "wins", "losses", "winrate",
                "base_winrate_no_model", "expectancy_usd", "total_pnl_usd", "avg_rr"]]
          .to_string(index=False))

    print(f"\n=== Hasil pada threshold default ({config.PROB_THRESHOLD}) ===")
    m = evaluate(res)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:24s}: {v:.4f}")
        else:
            print(f"  {k:24s}: {v}")
    print(f"\n  >> WINRATE (model, OOS)        : {m['winrate']*100:.1f}%")
    print(f"  >> Winrate tanpa model (semua) : {m['base_winrate_no_model']*100:.1f}%")

    # 6) Model final + sinyal terbaru
    final = fit_full_model(df)
    joblib.dump({"model": final, "features": FEATURE_COLS, "config": {
        "TP_POINTS": config.TP_POINTS, "SYMBOL": config.SYMBOL,
        "TIMEFRAME": config.TIMEFRAME_LABEL, "threshold": config.PROB_THRESHOLD,
    }}, "model_xauusd.joblib")
    print("\n[save] model final -> model_xauusd.joblib")

    res.to_csv("oos_trades.csv", index=False)
    print("[save] trade out-of-sample -> oos_trades.csv")

    # sinyal pada bar terbaru
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    p = final.predict_proba(latest[FEATURE_COLS].to_frame().T)[0, 1]
    print("\n=== Sinyal BAR TERBARU ===")
    print(f"  Waktu       : {latest['datetime']}")
    print(f"  Close       : {latest['close']:.2f}")
    print(f"  Support     : {latest['support']:.2f}" if pd.notna(latest['support']) else "  Support     : -")
    print(f"  P(TP dulu)  : {p:.3f}")
    if pd.notna(latest["support"]):
        sl_price = latest["support"] - config.SL_BUFFER_ATR * latest["atr"]
        sl_d = latest["close"] - sl_price
        tp_price = latest["close"] + config.TP_POINTS
        valid = config.MIN_SL_POINTS <= sl_d <= config.MAX_SL_POINTS
        action = "BUY" if (p >= config.PROB_THRESHOLD and valid) else "TUNGGU"
        print(f"  Entry       : {latest['close']:.2f}")
        print(f"  TP (tetap)  : {tp_price:.2f}  (+{config.TP_POINTS})")
        print(f"  SL (support): {sl_price:.2f}  (-{sl_d:.2f})  RR={config.TP_POINTS/sl_d:.2f}" if sl_d>0 else "  SL invalid")
        print(f"  >> AKSI     : {action}")
    return df, res, sweep


if __name__ == "__main__":
    import sys
    main(force_fetch="--fetch" in sys.argv)
