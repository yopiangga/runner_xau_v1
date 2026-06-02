"""Bandingkan performa model di timeframe berbeda (1h vs 5m) dengan
beberapa setting TP/horizon, lalu laporkan winrate out-of-sample.

Strategi tetap sama: BUY, TP tetap, SL variatif di support.
"""
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_klines
from features import build_features
from labeling import make_labels
from model import walk_forward, evaluate

pd.set_option("display.width", 160)

# (label, kline_type, total_bars, TP_POINTS, HOLD_HORIZON, MIN_SL, MAX_SL)
SCENARIOS = [
    ("1h  TP6  H24",  5, 8000,  6.0, 24, 1.5, 12.0),
    ("5m  TP6  H24",  2, 14000, 6.0, 24, 1.5, 12.0),   # param 1h dipaksa ke 5m
    ("5m  TP2  H48",  2, 14000, 2.0, 48, 0.6, 4.0),    # scalping menengah
    ("5m  TP1.5 H72", 2, 14000, 1.5, 72, 0.4, 3.0),    # scalping cepat
    ("5m  TP3  H96",  2, 14000, 3.0, 96, 0.8, 6.0),    # scalping lambat
]

TF_NAME = {2: "5m", 5: "1h"}


def run_scenario(name, kt, bars, tp, horizon, min_sl, max_sl):
    # set parameter ke config (dibaca runtime oleh modul lain)
    config.KLINE_TYPE = kt
    config.TIMEFRAME_LABEL = TF_NAME[kt]
    config.TOTAL_BARS = bars
    config.TP_POINTS = tp
    config.HOLD_HORIZON = horizon
    config.MIN_SL_POINTS = min_sl
    config.MAX_SL_POINTS = max_sl
    cache = f"data/{config.SYMBOL}_{TF_NAME[kt]}.csv"

    df = fetch_klines(kline_type=kt, total_bars=bars, cache_path=cache)
    span_days = (df["datetime"].max() - df["datetime"].min()).days
    df = build_features(df)
    df = make_labels(df)

    valid = df.dropna(subset=["label"])
    base_wr = (valid["label"] == 1).mean() * 100
    n_setup = len(valid)

    rows = []
    try:
        res, auc = walk_forward(df, verbose=False)
        for th in [0.60, 0.70, 0.80]:
            r = res.copy()
            r["signal"] = (r["proba"] >= th).astype(int)
            m = evaluate(r, th)
            rows.append({
                "skenario": name, "hari_data": span_days, "setup_valid": n_setup,
                "base_wr%": round(base_wr, 1), "auc": round(auc, 3),
                "thr": th, "n_trade": m["n_trades"],
                "winrate%": round(m["winrate"] * 100, 1) if m["n_trades"] else None,
                "exp_usd": round(m["expectancy_usd"], 2) if m["n_trades"] else None,
                "pnl_usd": round(m["total_pnl_usd"], 1) if m["n_trades"] else None,
                "avg_sl": round(valid["sl_dist"].mean(), 2),
            })
    except Exception as e:
        rows.append({"skenario": name, "hari_data": span_days,
                     "setup_valid": n_setup, "base_wr%": round(base_wr, 1),
                     "auc": None, "thr": None, "n_trade": 0,
                     "winrate%": None, "exp_usd": None, "pnl_usd": None,
                     "avg_sl": None})
        print(f"  [warn] {name}: {e}")
    return rows


def main():
    all_rows = []
    for sc in SCENARIOS:
        print(f"\n>>> Skenario: {sc[0]}")
        all_rows += run_scenario(*sc)
    out = pd.DataFrame(all_rows)
    print("\n" + "=" * 110)
    print("PERBANDINGAN TIMEFRAME (winrate out-of-sample)")
    print("=" * 110)
    print(out.to_string(index=False))
    out.to_csv("compare_timeframes.csv", index=False)
    print("\n[save] -> compare_timeframes.csv")


if __name__ == "__main__":
    main()
