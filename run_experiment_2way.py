"""Eksperimen DUA ARAH: BUY (long) + SELL (short).

- BUY : TP tetap di atas, SL variatif di bawah support.
- SELL: TP tetap di bawah, SL variatif di atas resistance.
Satu model memprediksi P(TP-dulu) dengan fitur 'side' (0=buy, 1=sell),
sehingga peluang trade ~2x lipat. Cocok bila sering trade tidak masalah.
"""
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import walk_forward, evaluate, threshold_sweep, fit_full_model

pd.set_option("display.width", 140)
FEATS = FEATURE_COLS  # sudah memuat 'side' lewat penambahan di bawah
FEATS_2WAY = FEATURE_COLS + ["side"]


def main(force_fetch=False):
    print("=" * 72)
    print(f"EKSPERIMEN 2-ARAH (BUY+SELL) — {config.SYMBOL} (XAUUSD) {config.TIMEFRAME_LABEL}")
    print(f"TP tetap = {config.TP_POINTS} | SL variatif (support/resistance) | "
          f"horizon = {config.HOLD_HORIZON} bar")
    print("=" * 72)

    df = fetch_klines(force=force_fetch)
    df = build_features(df)

    # Label kedua arah, lalu gabung jadi satu dataset bertimeline
    dfl = make_labels(df, side="long")
    dfs = make_labels(df, side="short")

    keep = ["datetime", "close", "support", "resistance", "atr"] + FEATURE_COLS + \
           ["label", "sl_dist", "tp_dist", "rr", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.dropna(subset=["label"])
    both = both.sort_values(["datetime", "side"]).reset_index(drop=True)

    # statistik dasar per arah
    for name, sd in [("BUY ", 0.0), ("SELL", 1.0)]:
        sub = both[both["side"] == sd]
        wr = (sub["label"] == 1).mean() * 100 if len(sub) else 0
        print(f"  setup {name}: {len(sub):5d}  base winrate {wr:.1f}%")
    print(f"  TOTAL setup : {len(both)}")

    print("\n=== Backtest Walk-Forward 2-arah (out-of-sample) ===")
    res, auc = walk_forward(both, n_splits=config.N_SPLITS, verbose=True,
                            feature_cols=FEATS_2WAY)
    print(f"AUC rata-rata: {auc:.3f}")

    print("\n=== Sweep Threshold (gabungan BUY+SELL, out-of-sample) ===")
    sweep = threshold_sweep(res)
    show = sweep.copy()
    show["winrate"] = (show["winrate"] * 100).round(1)
    print(show[["threshold", "n_trades", "wins", "losses", "winrate",
                "expectancy_usd", "total_pnl_usd"]].to_string(index=False))

    # rincian per arah pada threshold default
    print(f"\n=== Rincian per arah @ threshold {config.PROB_THRESHOLD} ===")
    sig = res[res["proba"] >= config.PROB_THRESHOLD]
    for name, sd in [("BUY ", 0.0), ("SELL", 1.0)]:
        s = sig[sig["side"] == sd]
        n = len(s); w = int((s["label"] == 1).sum())
        wr = w / n * 100 if n else 0
        print(f"  {name}: {n:4d} trade | winrate {wr:.1f}% | menang {w} kalah {n-w}")
    m = evaluate(res)
    print(f"  GABUNGAN: {m['n_trades']} trade | winrate {m['winrate']*100:.1f}% | "
          f"PnL {m['total_pnl_usd']:.1f} USD | exp {m['expectancy_usd']:.2f}/trade")

    # model final + simpan
    final = fit_full_model(both, feature_cols=FEATS_2WAY)
    joblib.dump({"model": final, "features": FEATS_2WAY, "two_way": True,
                 "config": {"TP_POINTS": config.TP_POINTS, "SYMBOL": config.SYMBOL,
                            "TIMEFRAME": config.TIMEFRAME_LABEL,
                            "threshold": config.PROB_THRESHOLD}},
                "model_xauusd_2way.joblib")
    res.to_csv("oos_trades_2way.csv", index=False)
    print("\n[save] model_xauusd_2way.joblib & oos_trades_2way.csv")
    return both, res, sweep


if __name__ == "__main__":
    import sys
    main(force_fetch="--fetch" in sys.argv)
