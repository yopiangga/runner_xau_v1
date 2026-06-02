"""Model klasifikasi (Gradient Boosting) + backtest walk-forward.

Ide untuk WINRATE TINGGI: model memprediksi probabilitas TP-dulu untuk
tiap setup. Kita hanya mengeksekusi BUY bila probabilitas >= threshold,
sehingga jumlah trade lebih sedikit tapi kualitas (winrate) lebih tinggi.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import config
from features import FEATURE_COLS


def _new_model():
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        l2_regularization=1.0,
        min_samples_leaf=40,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=config.RANDOM_STATE,
    )


def walk_forward(df, threshold=None, n_splits=None, verbose=True,
                 feature_cols=None, model_factory=None):
    """Backtest walk-forward (expanding window). Latih di masa lalu,
    uji di masa depan. Kembalikan dataframe trade pada periode uji.
    model_factory: fungsi tanpa argumen yang mengembalikan estimator baru."""
    threshold = config.PROB_THRESHOLD if threshold is None else threshold
    n_splits = n_splits or config.N_SPLITS
    FEATURE_COLS_ = feature_cols or FEATURE_COLS
    make = model_factory or _new_model

    data = df.dropna(subset=["label"] + FEATURE_COLS_).reset_index(drop=True)
    data = data.sort_values("datetime").reset_index(drop=True)
    N = len(data)
    if N < 300:
        raise RuntimeError(f"Data setup valid terlalu sedikit ({N}).")

    # expanding window: mulai latih dari 40% pertama
    start = int(N * 0.4)
    fold_edges = np.linspace(start, N, n_splits + 1, dtype=int)

    all_test = []
    aucs = []
    for k in range(n_splits):
        tr_end = fold_edges[k]
        te_end = fold_edges[k + 1]
        if te_end <= tr_end:
            continue
        train = data.iloc[:tr_end]
        test = data.iloc[tr_end:te_end].copy()
        if train["label"].nunique() < 2 or len(test) == 0:
            continue

        m = make()
        m.fit(train[FEATURE_COLS_], train["label"])
        proba = m.predict_proba(test[FEATURE_COLS_])[:, 1]
        test["proba"] = proba
        if test["label"].nunique() > 1:
            aucs.append(roc_auc_score(test["label"], proba))
        all_test.append(test)
        if verbose:
            print(f"  fold {k+1}: train={len(train)} test={len(test)} "
                  f"({test['datetime'].min().date()}→{test['datetime'].max().date()})")

    res = pd.concat(all_test).reset_index(drop=True)
    res["signal"] = (res["proba"] >= threshold).astype(int)
    return res, (np.mean(aucs) if aucs else float("nan"))


def evaluate(res, threshold=None):
    """Hitung metrik trading dari sinyal model (out-of-sample)."""
    threshold = config.PROB_THRESHOLD if threshold is None else threshold
    trades = res[res["signal"] == 1]
    n = len(trades)
    wins = int((trades["label"] == 1).sum())
    losses = int((trades["label"] == 0).sum())
    winrate = wins / n if n else float("nan")

    # PnL: menang -> +TP tetap; kalah -> -SL variatif
    pnl = np.where(trades["label"] == 1, trades["tp_dist"], -trades["sl_dist"])
    total_pnl = float(pnl.sum())
    expectancy = float(pnl.mean()) if n else float("nan")

    base_rate = (res["label"] == 1).mean()  # winrate jika ambil semua setup
    return {
        "threshold": threshold,
        "n_trades": n,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "base_winrate_no_model": float(base_rate),
        "expectancy_usd": expectancy,
        "total_pnl_usd": total_pnl,
        "avg_rr": float(trades["rr"].mean()) if n else float("nan"),
    }


def threshold_sweep(res, grid=None):
    grid = grid or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []
    for t in grid:
        r = res.copy()
        r["signal"] = (r["proba"] >= t).astype(int)
        m = evaluate(r, t)
        rows.append(m)
    return pd.DataFrame(rows)


def fit_full_model(df, feature_cols=None):
    """Latih model final pada SELURUH data (untuk dipakai prediksi live)."""
    FEATURE_COLS_ = feature_cols or FEATURE_COLS
    data = df.dropna(subset=["label"] + FEATURE_COLS_)
    m = _new_model()
    m.fit(data[FEATURE_COLS_], data["label"])
    return m
