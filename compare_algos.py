"""Bandingkan berbagai algoritma ML untuk sinyal BUY+SELL XAUUSD 5m.

Membandingkan secara ADIL: prediksi out-of-sample tiap model di-rank
berdasarkan probabilitas, lalu diambil top-K% paling yakin sebagai trade,
sehingga jumlah trade setara antar model. Dilaporkan winrate pada tiap
tingkat selektivitas + AUC. Model terbaik dipilih otomatis.
"""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from data_provider import fetch_klines
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import walk_forward

FEATS = FEATURE_COLS + ["side"]
RS = config.RANDOM_STATE


def sc(est):  # bungkus dengan scaler untuk model sensitif skala
    return make_pipeline(StandardScaler(), est)


ALGOS = {
    "LogisticRegression": lambda: sc(LogisticRegression(max_iter=1000, C=0.5)),
    "GaussianNB":         lambda: sc(GaussianNB()),
    "KNN":                lambda: sc(KNeighborsClassifier(n_neighbors=45)),
    "DecisionTree":       lambda: DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=RS),
    "RandomForest":       lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=RS),
    "ExtraTrees":         lambda: ExtraTreesClassifier(n_estimators=400, max_depth=10, min_samples_leaf=30, n_jobs=-1, random_state=RS),
    "AdaBoost":           lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=RS),
    "GradientBoosting":   lambda: GradientBoostingClassifier(max_depth=3, n_estimators=300, learning_rate=0.05, random_state=RS),
    "HistGradientBoost":  lambda: HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=400, l2_regularization=1.0, min_samples_leaf=40, early_stopping=True, validation_fraction=0.15, random_state=RS),
    "MLP":                lambda: sc(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=400, early_stopping=True, random_state=RS)),
}

# tingkat selektivitas: ambil top-X% sinyal paling yakin (per fold pooled)
TOP_FRACS = [0.10, 0.20, 0.40]


def build_dataset():
    df = fetch_klines()
    print(f"\n>>> DATA MENTAH: {len(df)} bar {config.TIMEFRAME_LABEL}")
    print(f"    dari {df['datetime'].min()}  s/d  {df['datetime'].max()}")
    df = build_features(df)
    dfl = make_labels(df, "long")
    dfs = make_labels(df, "short")
    keep = ["datetime"] + FEATURE_COLS + ["label", "sl_dist", "tp_dist", "rr", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], np.nan).dropna(subset=["label"] + FEATS)
    both = both.sort_values(["datetime", "side"]).reset_index(drop=True)
    return both


def report_periods(both):
    data = both.dropna(subset=["label"] + FEATS).sort_values("datetime").reset_index(drop=True)
    N = len(data)
    start = int(N * 0.4)
    edges = np.linspace(start, N, config.N_SPLITS + 1, dtype=int)
    print("\n" + "=" * 70)
    print("PERIODE DATA (bulan) — train, test, backtest")
    print("=" * 70)
    print(f"Total setup berlabel : {N}  ({data['datetime'].min()} → {data['datetime'].max()})")
    warm = data.iloc[:start]
    print(f"Warm-up train awal   : {warm['datetime'].min():%Y-%m-%d} → {warm['datetime'].max():%Y-%m-%d}  ({start} setup)")
    print("Walk-forward (expanding train, test di depannya):")
    for k in range(config.N_SPLITS):
        tr = data.iloc[:edges[k]]
        te = data.iloc[edges[k]:edges[k+1]]
        print(f"  fold {k+1}: TRAIN {tr['datetime'].min():%Y-%m-%d} → {tr['datetime'].max():%Y-%m-%d}"
              f"  | TEST/BACKTEST {te['datetime'].min():%Y-%m-%d} → {te['datetime'].max():%Y-%m-%d}")
    print("Catatan: backtest = gabungan seluruh periode TEST (out-of-sample).")


def winrate_topk(res, frac):
    n = max(1, int(len(res) * frac))
    top = res.nlargest(n, "proba")
    w = int((top["label"] == 1).sum())
    pnl = np.where(top["label"] == 1, top["tp_dist"], -top["sl_dist"]).sum()
    return len(top), w / len(top) * 100, float(pnl)


def main():
    from sklearn.metrics import roc_auc_score
    both = build_dataset()
    report_periods(both)

    print("\n" + "=" * 70)
    print("PERBANDINGAN ALGORITMA (out-of-sample, BUY+SELL)")
    print("=" * 70)
    rows = []
    for name, fac in ALGOS.items():
        try:
            res, auc = walk_forward(both, verbose=False, feature_cols=FEATS, model_factory=fac)
            row = {"algoritma": name, "auc": round(auc, 3)}
            for f in TOP_FRACS:
                n, wr, pnl = winrate_topk(res, f)
                row[f"WR@top{int(f*100)}%"] = round(wr, 1)
                row[f"n@{int(f*100)}%"] = n
            rows.append(row)
            print(f"  {name:18s} selesai (AUC {auc:.3f})")
        except Exception as e:
            print(f"  {name:18s} GAGAL: {e}")
    out = pd.DataFrame(rows)

    print("\n--- AUC (kualitas pemeringkatan sinyal; makin tinggi makin baik) ---")
    print(out[["algoritma", "auc"]].sort_values("auc", ascending=False).to_string(index=False))

    print("\n--- WINRATE pada tingkat selektivitas sama (jumlah trade setara) ---")
    cols = ["algoritma"] + [c for c in out.columns if c.startswith("WR@")]
    print(out[cols].sort_values("WR@top20%", ascending=False).to_string(index=False))

    best = out.sort_values(["WR@top20%", "auc"], ascending=False).iloc[0]
    print("\n" + "=" * 70)
    print(f"TERBAIK: {best['algoritma']}  | AUC {best['auc']}  "
          f"| WR@top20% {best['WR@top20%']}%  | WR@top10% {best['WR@top10%']}%")
    print("=" * 70)
    out.to_csv("compare_algos.csv", index=False)
    print("[save] -> compare_algos.csv")


if __name__ == "__main__":
    main()
