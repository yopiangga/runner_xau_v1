"""Report backtest detail: posisi BELI, TP, SL, hasil, dan apakah SESUAI sinyal.

- Entry  = harga close saat sinyal BUY muncul
- TP      = entry + TP_POINTS (tetap)
- SL      = support - buffer*ATR (variatif)
- Sinyal  = model memprediksi P(TP dulu) >= threshold  -> AKSI BUY
- SESUAI? = sinyal BUY memang berakhir MENANG (TP). TIDAK SESUAI bila kena SL.
"""
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import config

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)


def build_report(threshold=None, save=True):
    threshold = config.PROB_THRESHOLD if threshold is None else threshold
    df = pd.read_csv("oos_trades.csv", parse_dates=["datetime"])

    # hanya bar yang menghasilkan SINYAL BUY (proba >= threshold)
    t = df[df["proba"] >= threshold].copy().reset_index(drop=True)

    t["entry"] = t["close"].round(2)
    t["tp_price"] = (t["close"] + t["tp_dist"]).round(2)
    t["sl_price"] = (t["close"] - t["sl_dist"]).round(2)
    t["tp_jarak"] = t["tp_dist"].round(2)
    t["sl_jarak"] = t["sl_dist"].round(2)
    t["rr"] = t["rr"].round(2)
    t["prob_menang"] = (t["proba"] * 100).round(1)
    t["hasil"] = t["label"].map({1: "MENANG (TP)", 0: "KALAH (SL)"})
    # Sinyal selalu BUY (karena lolos threshold). SESUAI bila benar-benar menang.
    t["sinyal"] = "BUY"
    t["sesuai_sinyal"] = t["label"].map({1: "SESUAI", 0: "TIDAK SESUAI"})
    t["pnl"] = t.apply(lambda r: r["tp_dist"] if r["label"] == 1 else -r["sl_dist"], axis=1).round(2)
    t["ekuitas"] = t["pnl"].cumsum().round(2)

    cols = ["datetime", "sinyal", "entry", "tp_price", "sl_price", "tp_jarak",
            "sl_jarak", "rr", "prob_menang", "hasil", "sesuai_sinyal", "pnl", "ekuitas"]
    report = t[cols]

    # ---- Ringkasan ----
    n = len(report)
    menang = int((t["label"] == 1).sum())
    kalah = int((t["label"] == 0).sum())
    winrate = menang / n * 100 if n else 0
    sesuai = menang
    tidak = kalah
    print("=" * 100)
    print(f"REPORT BACKTEST — BUY {config.SYMBOL} (XAUUSD) {config.TIMEFRAME_LABEL} | threshold sinyal = {threshold}")
    print("=" * 100)
    print(f"Total sinyal BUY        : {n}")
    print(f"SESUAI sinyal (menang)  : {sesuai}  ({sesuai/n*100:.1f}%)")
    print(f"TIDAK SESUAI (kalah)    : {tidak}  ({tidak/n*100:.1f}%)")
    print(f">> WINRATE              : {winrate:.1f}%")
    print(f"TP tetap                : {config.TP_POINTS} USD")
    print(f"SL rata-rata (variatif) : {t['sl_dist'].mean():.2f} USD  (min {t['sl_dist'].min():.2f} / max {t['sl_dist'].max():.2f})")
    print(f"RR rata-rata            : {t['rr'].mean():.2f}")
    print(f"Total PnL               : {t['pnl'].sum():.2f} USD")
    print(f"Ekspektansi per trade   : {t['pnl'].mean():.2f} USD")
    print(f"Periode                 : {t['datetime'].min()} → {t['datetime'].max()}")

    print("\n" + "-" * 100)
    print("DETAIL TIAP POSISI BUY")
    print("-" * 100)
    show = report.copy()
    show["datetime"] = show["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False))

    if save:
        report.to_csv("report_backtest.csv", index=False)
        print(f"\n[save] report lengkap -> report_backtest.csv ({n} baris)")
    return report


if __name__ == "__main__":
    import sys
    th = float(sys.argv[1]) if len(sys.argv) > 1 else None
    build_report(th)
