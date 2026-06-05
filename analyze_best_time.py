"""Evaluasi WAKTU TERBAIK untuk trade (XAUUSD 5m).

Menjawab: jam/sesi/hari mana yang paling menguntungkan untuk mengeksekusi
sinyal model. Dinilai bukan hanya dari WINRATE, tapi juga EKSPEKTANSI (PnL
rata-rata per trade) karena RR < 1 (TP tetap $2, SL variatif) — winrate tinggi
belum tentu profit.

Sumber sinyal: out-of-sample walk-forward dari model TERBAIK (config.MODEL_ALGO),
sehingga hasilnya tidak bias look-ahead. Waktu dikonversi UTC -> WIB (UTC+7).

  .venv/bin/python analyze_best_time.py
  .venv/bin/python analyze_best_time.py --threshold 0.7
"""
import sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import walk_forward

FEATS = FEATURE_COLS + ["side"]
WIB_OFFSET = pd.Timedelta(hours=7)   # data AllTick = UTC; WIB = UTC+7


def sesi(h):
    if 14 <= h < 19: return "1. London (14-19)"
    if 19 <= h < 24: return "2. London+NY overlap (19-24)"
    if 0 <= h < 5:   return "3. New York (00-05)"
    return "4. Asia/sepi (05-14)"


NAMA_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


def build_oos(threshold):
    """Bangun trade out-of-sample dari walk-forward model terbaik."""
    df = build_features(fetch_klines())
    dfl = make_labels(df, "long"); dfs = make_labels(df, "short")
    keep = ["datetime"] + FEATURE_COLS + ["label", "sl_dist", "tp_dist", "rr", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"] + FEATS)
    both = both.sort_values(["datetime", "side"]).reset_index(drop=True)

    res, auc = walk_forward(both, threshold=threshold, verbose=False, feature_cols=FEATS)
    sig = res[res["proba"] >= threshold].copy()
    # PnL per trade: menang -> +TP tetap, kalah -> -SL variatif
    sig["pnl"] = np.where(sig["label"] == 1, sig["tp_dist"], -sig["sl_dist"])
    sig["menang"] = (sig["label"] == 1).astype(int)
    wib = sig["datetime"] + WIB_OFFSET
    sig["jam"] = wib.dt.hour
    sig["hari"] = wib.dt.dayofweek
    sig["sesi"] = sig["jam"].apply(sesi)
    return sig, auc


def ringkas(g):
    n = len(g)
    wr = g["menang"].mean() * 100
    exp = g["pnl"].mean()
    return pd.Series({
        "sinyal": n,
        "winrate%": round(wr, 1),
        "ekspektansi$": round(exp, 3),
        "total_pnl$": round(g["pnl"].sum(), 1),
    })


def main():
    th = config.PROB_THRESHOLD
    if "--threshold" in sys.argv:
        th = float(sys.argv[sys.argv.index("--threshold") + 1])

    print("=" * 68)
    print(f"  EVALUASI WAKTU TERBAIK — {config.SYMBOL} {config.TIMEFRAME_LABEL} "
          f"| model {config.MODEL_ALGO} | threshold {th:.0%}")
    print("=" * 68)
    sig, auc = build_oos(th)
    print(f"OOS: {len(sig)} sinyal (AUC {auc:.3f}) | "
          f"{sig['datetime'].min():%Y-%m-%d} → {sig['datetime'].max():%Y-%m-%d} | "
          f"winrate total {sig['menang'].mean()*100:.1f}% | "
          f"ekspektansi total ${sig['pnl'].mean():.3f}/trade\n")

    # --- Per JAM (WIB) ---
    per_jam = sig.groupby("jam").apply(ringkas).reset_index()
    per_jam["jam"] = per_jam["jam"].apply(lambda h: f"{int(h):02d}:xx")
    per_jam["bar"] = (per_jam["winrate%"]).apply(lambda w: "█" * int(w / 5))
    print("--- PER JAM (WIB) ---  (ekspektansi$ = profit rata-rata per trade)")
    print(per_jam.to_string(index=False))

    # --- Per SESI ---
    per_sesi = sig.groupby("sesi").apply(ringkas).reset_index().sort_values("sesi")
    print("\n--- PER SESI (WIB) ---")
    print(per_sesi.to_string(index=False))

    # --- Per HARI ---
    per_hari = sig.groupby("hari").apply(ringkas).reset_index()
    per_hari["hari"] = per_hari["hari"].apply(lambda d: NAMA_HARI[int(d)])
    print("\n--- PER HARI ---")
    print(per_hari.to_string(index=False))

    # --- Rekomendasi: jam dgn ekspektansi POSITIF & sampel cukup ---
    MIN_N = max(10, int(len(sig) * 0.02))
    pj = sig.groupby("jam").apply(ringkas).reset_index()
    bagus = pj[(pj["ekspektansi$"] > 0) & (pj["sinyal"] >= MIN_N)].sort_values(
        "ekspektansi$", ascending=False)
    print("\n" + "=" * 68)
    print(f"REKOMENDASI — jam (WIB) dgn ekspektansi POSITIF & >= {MIN_N} sinyal:")
    print("=" * 68)
    if len(bagus):
        for _, r in bagus.iterrows():
            print(f"  {int(r['jam']):02d}:xx WIB | {int(r['sinyal']):3d} sinyal | "
                  f"WR {r['winrate%']:.0f}% | ekspektansi ${r['ekspektansi$']:.3f}/trade "
                  f"| total ${r['total_pnl$']:.1f}")
        best_sesi = per_sesi.sort_values("ekspektansi$", ascending=False).iloc[0]
        print(f"\nSesi terbaik: {best_sesi['sesi']}  "
              f"(WR {best_sesi['winrate%']:.0f}%, ekspektansi ${best_sesi['ekspektansi$']:.3f}/trade)")
    else:
        print("  Tidak ada jam dengan ekspektansi positif & sampel memadai pada threshold ini.")

    per_jam.to_csv("best_time_per_hour.csv", index=False)
    print("\n[save] -> best_time_per_hour.csv")


if __name__ == "__main__":
    main()
