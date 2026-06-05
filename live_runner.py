"""Runner OTOMATIS: jalan terus, cek tiap penutupan candle 5 menit,
kirim NOTIFIKASI (macOS) saat ada sinyal BUY/SELL.

  python3 live_runner.py            # jalan terus, cek tiap candle 5m
  python3 live_runner.py --once     # cek sekali lalu keluar (untuk uji)
  python3 live_runner.py --interval 60   # paksa cek tiap 60 detik
  python  live_runner.py --trade    # + AUTO BUKA POSISI via MT5 (Windows)
  python  live_runner.py --no-trade # paksa MATIKAN auto-trade (override .env)

Catatan:
- Model dilatih SEKALI saat start dari data historis (cache/AllTick),
  lalu dipakai ulang. Otomatis dilatih ulang tiap RETRAIN_HOURS jam.
- Tiap candle tutup -> fetch ringan bar terbaru -> prediksi -> notif.
- Anti-spam: 1 notifikasi per candle.
"""
import sys, time, subprocess, platform, warnings, datetime as dt
import ssl, urllib.request, urllib.parse, urllib.error
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import config
from data_provider import update_cache, fetch_recent
from features import build_features, FEATURE_COLS
from labeling import make_labels
from model import fit_full_model
import mt5_trader

FEATS = FEATURE_COLS + ["side"]
RETRAIN_HOURS = config.RETRAIN_HOURS   # latih ulang model tiap N jam
BUFFER_SEC = 20             # tunggu 20 detik setelah candle tutup (data settle)
RECENT_BARS = 400           # jumlah bar untuk hitung indikator live
LOG_FILE = "signals_log.csv"


# ---------- Notifikasi ----------
# Banner macOS sering diblokir izin -> pakai POPUP dialog yang pasti muncul,
# plus bunyi. Set POPUP=False kalau hanya mau banner + bunyi.
POPUP = True
POPUP_TIMEOUT = 20          # popup auto-tutup setelah 20 detik

def _ssl_context():
    """Konteks SSL dengan CA bundle certifi bila ada (atasi 'certificate verify
    failed' yang umum di Windows karena root CA Python tidak terpasang)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def send_telegram(text):
    """Kirim pesan ke Telegram (stdlib, tanpa dependency wajib). Diam jika belum dikonfigurasi."""
    token = config.TELEGRAM_BOT_TOKEN
    chat_id = config.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        with urllib.request.urlopen(url, data=data, timeout=10, context=_ssl_context()) as r:
            r.read()
        return
    except urllib.error.URLError as e:
        # Fallback Windows: root CA tak tersedia -> kirim tanpa verifikasi sertifikat.
        if isinstance(getattr(e, "reason", None), ssl.SSLError):
            print("  [warn] verifikasi SSL gagal; kirim tanpa verifikasi "
                  "(disarankan `pip install certifi` untuk perbaikan permanen).")
            try:
                with urllib.request.urlopen(
                    url, data=data, timeout=10, context=ssl._create_unverified_context()
                ) as r:
                    r.read()
                return
            except Exception as e2:
                print(f"  [warn] kirim telegram gagal: {e2}")
                return
        print(f"  [warn] kirim telegram gagal: {e}")
    except Exception as e:
        print(f"  [warn] kirim telegram gagal: {e}")


def notify(title, message, sound=True):
    print(f"\a🔔 {title} — {message}")
    # Telegram lintas-platform (jalan di Windows/Mac/Linux selama token & chat_id diisi)
    send_telegram(f"🔔 {title}\n{message}")
    if platform.system() != "Darwin":
        return
    msg = message.replace('"', "'")
    ttl = title.replace('"', "'")
    # 1) bunyi (selalu jalan)
    if sound:
        try:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Glass.aiff"])
        except Exception:
            pass
    # 2) banner (muncul jika izin diberi)
    try:
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "{ttl}"'],
                       check=False, timeout=5)
    except Exception:
        pass
    # 3) popup dialog (andal, tak butuh izin, auto-tutup)
    if POPUP:
        try:
            subprocess.run(["osascript", "-e",
                f'display dialog "{msg}" with title "{ttl}" buttons {{"OK"}} '
                f'default button "OK" giving up after {POPUP_TIMEOUT} with icon note'],
                check=False, timeout=POPUP_TIMEOUT + 5)
        except Exception as e:
            print(f"  [warn] popup gagal: {e}")


# ---------- Model ----------
def train_model():
    print(f"[{dt.datetime.now():%H:%M:%S}] update data + melatih ulang model...")
    df = update_cache()                       # Opsi B: sambung candle baru -> data fresh
    df = build_features(df)
    dfl = make_labels(df, "long"); dfs = make_labels(df, "short")
    keep = FEATURE_COLS + ["label", "sl_dist", "tp_dist", "side"]
    both = pd.concat([dfl[keep], dfs[keep]], ignore_index=True)
    both = both.replace([np.inf, -np.inf], np.nan)
    model = fit_full_model(both, feature_cols=FEATS)
    print(f"[{dt.datetime.now():%H:%M:%S}] model siap "
          f"({both.dropna(subset=['label']).shape[0]} setup latih).")
    return model


# ---------- Evaluasi sinyal pada candle terakhir ----------
def evaluate_latest(model):
    df = fetch_recent(num=RECENT_BARS)
    df = build_features(df)
    last = df.iloc[-1]
    close = last["close"]; atr = last["atr"]
    buf = config.SL_BUFFER_ATR * atr
    opts = []
    if pd.notna(last["support"]):
        sl_d = close - (last["support"] - buf)
        if config.MIN_SL_POINTS <= sl_d <= config.MAX_SL_POINTS:
            row = last[FEATURE_COLS].copy(); row["side"] = 0.0
            p = model.predict_proba(row.to_frame().T[FEATS])[0, 1]
            opts.append(("BUY", p, close + config.TP_POINTS, last["support"] - buf, sl_d))
    if pd.notna(last["resistance"]):
        sl_d = (last["resistance"] + buf) - close
        if config.MIN_SL_POINTS <= sl_d <= config.MAX_SL_POINTS:
            row = last[FEATURE_COLS].copy(); row["side"] = 1.0
            p = model.predict_proba(row.to_frame().T[FEATS])[0, 1]
            opts.append(("SELL", p, close - config.TP_POINTS, last["resistance"] + buf, sl_d))
    opts.sort(key=lambda x: x[1], reverse=True)
    return last, opts


def log_signal(candle_time, close, side, p, tp, sl, action):
    row = pd.DataFrame([{
        "waktu_cek": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candle": candle_time, "close": round(close, 2), "sinyal": side,
        "keyakinan": round(p * 100, 1), "tp": round(tp, 2),
        "sl": round(sl, 2), "aksi": action,
    }])
    import os
    row.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def auto_open(side, p, tp, sl, last):
    """Buka posisi otomatis via MT5 sesuai sinyal (hanya Windows + AUTO_TRADE)."""
    open_count = mt5_trader.count_open_positions()
    if open_count >= config.MAX_OPEN_POSITIONS:
        print(f"   [trade] lewati: sudah {open_count} posisi terbuka "
              f"(maks {config.MAX_OPEN_POSITIONS}).")
        return
    result, err = mt5_trader.open_position(side, sl, tp)
    if err:
        print(f"   [trade] GAGAL: {err}")
        send_telegram(f"⚠️ Auto-trade GAGAL {config.SYMBOL} {side}: {err}")
        return
    vol = getattr(result, "volume", config.TRADE_LOT)
    px = getattr(result, "price", last["close"])
    ticket = getattr(result, "order", "?")
    print(f"   [trade] ✅ OPEN {side} {vol} lot @ {px} (ticket {ticket})")
    send_telegram(
        f"✅ AUTO OPEN {config.SYMBOL} {side} {vol} lot @ {px:.2f}\n"
        f"TP {tp:.2f} | SL {sl:.2f} | yakin {p*100:.0f}% | ticket {ticket}"
    )


def run_cycle(model, state, auto_trade=False):
    try:
        last, opts = evaluate_latest(model)
    except Exception as e:
        print(f"[{dt.datetime.now():%H:%M:%S}] [warn] gagal cek: {e}")
        return
    candle = str(last["datetime"])
    if candle == state.get("last_candle"):
        return                                  # candle sama, sudah diproses
    state["last_candle"] = candle
    th = config.PROB_THRESHOLD
    now = dt.datetime.now().strftime("%H:%M:%S")

    if not opts:
        print(f"[{now}] candle {candle} close {last['close']:.2f} → TUNGGU (tak ada setup)")
        return
    side, p, tp, sl, sl_d = opts[0]
    if p >= th:
        action = side
        msg = (f"{side} @~{last['close']:.2f} | TP {tp:.2f} | SL {sl:.2f} "
               f"({sl_d:.2f}) | yakin {p*100:.0f}%")
        notify(f"SINYAL {config.SYMBOL} {side}", msg)
        print(f"[{now}] candle {candle} → 🔔 {msg}")
        if auto_trade:
            auto_open(side, p, tp, sl, last)
    else:
        action = "TUNGGU"
        print(f"[{now}] candle {candle} close {last['close']:.2f} → TUNGGU "
              f"(terkuat {side} {p*100:.0f}% < {th*100:.0f}%)")
    log_signal(candle, last["close"], side, p, tp, sl, action)


def seconds_to_next_candle(tf_min=5):
    now = time.time()
    return tf_min * 60 - (now % (tf_min * 60)) + BUFFER_SEC


def main():
    once = "--once" in sys.argv
    interval = None
    if "--interval" in sys.argv:
        interval = int(sys.argv[sys.argv.index("--interval") + 1])

    # Auto-trade: default dari .env (AUTO_TRADE), bisa di-override via CLI.
    auto_trade = config.AUTO_TRADE
    if "--trade" in sys.argv:
        auto_trade = True
    if "--no-trade" in sys.argv:
        auto_trade = False
    if auto_trade and not mt5_trader.is_available():
        print("[warn] auto-trade diminta tapi MT5 tidak tersedia "
              "(hanya Windows + paket MetaTrader5). Mode NOTIFIKASI saja.")
        auto_trade = False

    print("=" * 60)
    print(f"  LIVE RUNNER {config.SYMBOL} (XAUUSD) {config.TIMEFRAME_LABEL}")
    print(f"  threshold {config.PROB_THRESHOLD:.0%} | notif macOS | log → {LOG_FILE}")
    if auto_trade:
        print(f"  🤖 AUTO-TRADE AKTIF | lot {config.TRADE_LOT} | "
              f"maks {config.MAX_OPEN_POSITIONS} posisi | magic {config.TRADE_MAGIC}")
    print("=" * 60)
    model = train_model()
    last_train = time.time()
    state = {}

    send_telegram(
        f"🚀 Bot {config.SYMBOL} (XAUUSD) {config.TIMEFRAME_LABEL} START\n"
        f"Sumber data: {config.DATA_PROVIDER} | threshold {config.PROB_THRESHOLD:.0%}\n"
        f"Mode: {'sekali (--once)' if once else 'live tiap candle'} | "
        f"{'🤖 AUTO-TRADE ON' if auto_trade else 'notif saja'} | "
        f"waktu {dt.datetime.now():%Y-%m-%d %H:%M:%S}"
    )

    if once:
        run_cycle(model, state, auto_trade)
        return

    print("Berjalan... (Ctrl+C untuk berhenti)\n")
    try:
        while True:
            wait = interval if interval else seconds_to_next_candle()
            time.sleep(max(1, wait))
            if time.time() - last_train > RETRAIN_HOURS * 3600:
                model = train_model(); last_train = time.time()
            run_cycle(model, state, auto_trade)
    except KeyboardInterrupt:
        print("\nDihentikan.")


if __name__ == "__main__":
    main()
