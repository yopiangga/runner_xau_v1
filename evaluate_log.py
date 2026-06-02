"""Evaluasi signals_log.csv: cek tiap sinyal kena TP atau SL duluan
berdasarkan harga AKTUAL setelah sinyal -> tandai BENAR / SALAH / BELUM."""
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import fetch_recent

HORIZON = config.HOLD_HORIZON


def evaluate(only_action=True, min_conf=None):
    log = pd.read_csv("signals_log.csv", parse_dates=["candle"])
    if min_conf is not None:
        log = log[log["keyakinan"] >= min_conf].copy()
    elif only_action:
        log = log[log["aksi"].isin(["BUY", "SELL"])].copy()
    log = log.reset_index(drop=True)

    px = fetch_recent(num=500)                       # data 5m terbaru (aktual)
    px = px.sort_values("datetime").reset_index(drop=True)
    t = px["datetime"].values
    hi = px["high"].values; lo = px["low"].values; cl = px["close"].values

    rows = []
    for _, s in log.iterrows():
        side = s["sinyal"]; tp = s["tp"]; sl = s["sl"]
        # cari index candle sinyal
        match = np.where(px["datetime"] == s["candle"])[0]
        if len(match) == 0:
            rows.append({**base(s), "hasil": "DATA TAK ADA", "pnl": None}); continue
        i = int(match[0])
        hasil, pnl, kapan = "BELUM SELESAI", None, None
        end = min(len(px), i + 1 + HORIZON)
        for j in range(i + 1, end):
            if side == "SELL":
                hit_tp = lo[j] <= tp; hit_sl = hi[j] >= sl
            else:
                hit_tp = hi[j] >= tp; hit_sl = lo[j] <= sl
            if hit_sl and hit_tp:
                hasil, pnl, kapan = "SALAH (SL)", -abs(s["close"] - sl), px["datetime"].iloc[j]; break
            if hit_sl:
                hasil, pnl, kapan = "SALAH (SL)", -abs(s["close"] - sl), px["datetime"].iloc[j]; break
            if hit_tp:
                hasil, pnl, kapan = "BENAR (TP)", abs(s["close"] - tp), px["datetime"].iloc[j]; break
        rows.append({**base(s), "hasil": hasil,
                     "tutup": kapan, "pnl": round(pnl, 2) if pnl is not None else None})
    return pd.DataFrame(rows)


def base(s):
    return {"candle": s["candle"], "sinyal": s["sinyal"], "entry": s["close"],
            "tp": s["tp"], "sl": s["sl"], "keyakinan": s["keyakinan"]}


def main(only_action=True, min_conf=None):
    out = evaluate(only_action, min_conf)
    label = (f"keyakinan >= {min_conf}%" if min_conf is not None
             else "hanya yang dieksekusi" if only_action else "semua baris log")
    print("=" * 92)
    print(f"EVALUASI SINYAL  ({label})")
    print("=" * 92)
    show = out.copy()
    show["candle"] = pd.to_datetime(show["candle"]).dt.strftime("%m-%d %H:%M")
    if "tutup" in show:
        show["tutup"] = pd.to_datetime(show["tutup"]).dt.strftime("%H:%M")
    print(show.to_string(index=False))

    done = out[out["hasil"].str.startswith(("BENAR", "SALAH"))]
    benar = (out["hasil"] == "BENAR (TP)").sum()
    salah = out["hasil"].str.startswith("SALAH").sum()
    belum = (out["hasil"] == "BELUM SELESAI").sum()
    print("\n" + "-" * 60)
    print(f"Total sinyal      : {len(out)}")
    print(f"BENAR (kena TP)   : {benar}")
    print(f"SALAH (kena SL)   : {salah}")
    print(f"BELUM SELESAI     : {belum}")
    if len(done):
        print(f">> WINRATE (selesai): {benar/len(done)*100:.1f}%  ({benar}/{len(done)})")
        print(f">> PnL bersih(level): {done['pnl'].sum():.2f} USD")
    out.to_csv("signals_evaluated.csv", index=False)
    print("\n[save] signals_evaluated.csv")


if __name__ == "__main__":
    import sys
    mc = None
    if "--conf" in sys.argv:
        mc = float(sys.argv[sys.argv.index("--conf") + 1])
    main(only_action="--all" not in sys.argv, min_conf=mc)
