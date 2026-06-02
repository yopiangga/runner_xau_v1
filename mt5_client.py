"""Klien MetaTrader5 (MT5): ambil data XAUUSD via terminal MT5 (khusus Windows).

Antarmuka SAMA dengan alltick_client (fetch_recent / update_cache / fetch_klines)
sehingga bisa dipakai bergantian lewat data_provider berdasarkan DATA_PROVIDER.

Prasyarat (Windows):
  1. pip install MetaTrader5
  2. Terminal MetaTrader 5 ter-install dan login ke akun broker (boleh akun demo).
  3. Set di .env:
       DATA_PROVIDER=mt5
       MT5_SYMBOL=XAUUSD          # sesuaikan nama simbol di broker Anda
       MT5_LOGIN=...              # opsional (jika ingin login otomatis)
       MT5_PASSWORD=...           # opsional
       MT5_SERVER=...             # opsional
       MT5_PATH=C:\\...\\terminal64.exe   # opsional (jika terminal tak terdeteksi)

Catatan timezone: MT5 mengembalikan waktu bar dalam zona waktu SERVER broker
(sering UTC+2/UTC+3), bukan UTC murni seperti AllTick. Selama dipakai konsisten
(latih & live sama-sama dari MT5), model tidak terpengaruh. Jika menukar cache
AllTick <-> MT5, gunakan file cache terpisah (lihat MT5_CACHE_PATH di config).
"""
import os
import warnings
import pandas as pd
import config

warnings.filterwarnings("ignore")

try:
    import MetaTrader5 as mt5
except ImportError:  # bukan Windows / paket belum diinstall
    mt5 = None


# AllTick kline_type -> MT5 timeframe (agar param antarmuka tetap kompatibel).
# 1=1m 2=5m 3=15m 4=30m 5=1h 6=2h 7=4h 8=1d
def _tf(kline_type):
    table = {
        1: mt5.TIMEFRAME_M1, 2: mt5.TIMEFRAME_M5, 3: mt5.TIMEFRAME_M15,
        4: mt5.TIMEFRAME_M30, 5: mt5.TIMEFRAME_H1, 6: mt5.TIMEFRAME_H2,
        7: mt5.TIMEFRAME_H4, 8: mt5.TIMEFRAME_D1,
    }
    kt = kline_type or config.KLINE_TYPE
    if kt not in table:
        raise ValueError(f"kline_type {kt} tidak didukung untuk MT5.")
    return table[kt]


_INITIALIZED = False


def _ensure_init():
    """Inisialisasi koneksi ke terminal MT5 (sekali saja)."""
    global _INITIALIZED
    if mt5 is None:
        raise RuntimeError(
            "Paket MetaTrader5 tidak tersedia. Jalankan `pip install MetaTrader5` "
            "di Windows dengan terminal MetaTrader 5 ter-install."
        )
    if _INITIALIZED:
        return
    kwargs = {}
    if config.MT5_PATH:
        kwargs["path"] = config.MT5_PATH
    if config.MT5_LOGIN and config.MT5_PASSWORD and config.MT5_SERVER:
        kwargs["login"] = int(config.MT5_LOGIN)
        kwargs["password"] = config.MT5_PASSWORD
        kwargs["server"] = config.MT5_SERVER
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"mt5.initialize() gagal: {mt5.last_error()}")
    _INITIALIZED = True


def _symbol(code=None):
    """Pastikan simbol ada & tampil di Market Watch, kembalikan namanya."""
    sym = code or config.MT5_SYMBOL
    info = mt5.symbol_info(sym)
    if info is None:
        raise RuntimeError(
            f"Simbol '{sym}' tidak ditemukan di MT5. Cek nama simbol di broker "
            f"(mis. XAUUSD, XAUUSDm, GOLD) lalu set MT5_SYMBOL di .env."
        )
    if not info.visible:
        mt5.symbol_select(sym, True)
    return sym


def _rates_to_df(rates):
    """Ubah array rates MT5 -> DataFrame dengan skema yang sama seperti AllTick."""
    cols = ["timestamp", "datetime", "open", "high", "low", "close", "volume"]
    if rates is None or len(rates) == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rates)
    out = pd.DataFrame({
        "timestamp": df["time"].astype("int64"),
        "datetime": pd.to_datetime(df["time"], unit="s"),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        # MT5: tick_volume = jumlah tick (proxy volume); paling andal lintas broker
        "volume": df["tick_volume"].astype(float),
    })
    return out.sort_values("datetime").reset_index(drop=True)


def fetch_recent(code=None, kline_type=None, num=400):
    """Fetch RINGAN: `num` bar terbaru (untuk pemakaian live). Tidak menulis cache."""
    _ensure_init()
    sym = _symbol(code)
    rates = mt5.copy_rates_from_pos(sym, _tf(kline_type), 0, num)
    return _rates_to_df(rates)


def fetch_klines(code=None, kline_type=None, total_bars=None,
                 cache_path=None, force=False):
    """Ambil `total_bars` bar historis dari MT5. Hasil di-cache ke CSV."""
    code = code or config.MT5_SYMBOL
    total_bars = total_bars or config.TOTAL_BARS
    cache_path = cache_path or config.MT5_CACHE_PATH

    if cache_path and os.path.exists(cache_path) and not force:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        print(f"[cache] dimuat {len(df)} bar dari {cache_path}")
        return df

    _ensure_init()
    sym = _symbol(code)
    print(f"[fetch] mengambil ~{total_bars} bar {sym} (kline_type="
          f"{kline_type or config.KLINE_TYPE}) dari MT5...")
    rates = mt5.copy_rates_from_pos(sym, _tf(kline_type), 0, total_bars)
    df = _rates_to_df(rates)
    if len(df) < total_bars:
        print(f"  [warn] hanya {len(df)} bar tersedia dari MT5 (diminta {total_bars}). "
              f"Geser scroll grafik MT5 ke belakang agar histori ter-download.")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[cache] disimpan {len(df)} bar ke {cache_path}")
    return df


def update_cache(code=None, kline_type=None, cache_path=None, recent_num=1000):
    """Sambung bar BARU ke cache lalu kembalikan data ter-update (mirip Opsi B AllTick)."""
    code = code or config.MT5_SYMBOL
    cache_path = cache_path or config.MT5_CACHE_PATH

    if not os.path.exists(cache_path):
        return fetch_klines(code, kline_type, cache_path=cache_path)

    old = pd.read_csv(cache_path, parse_dates=["datetime"])
    new = fetch_recent(code, kline_type, num=recent_num)

    last_old = int(old["timestamp"].max())
    oldest_new = int(new["timestamp"].min())
    if oldest_new > last_old + 1:
        print(f"  [warn] ada celah data: cache s/d {last_old}, data baru mulai "
              f"{oldest_new}. Pertimbangkan force fetch penuh.")

    merged = pd.concat([old, new], ignore_index=True)
    # buang duplikat timestamp, simpan nilai TERBARU (bar terakhir bisa berubah)
    merged = merged.drop_duplicates(subset="timestamp", keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    added = len(merged) - len(old)
    merged.to_csv(cache_path, index=False)
    print(f"[update] cache: {len(old)} -> {len(merged)} bar (+{added} baru), "
          f"terbaru {merged['datetime'].iloc[-1]}")
    return merged


if __name__ == "__main__":
    df = fetch_klines()
    print(df.tail())
    print("Rentang:", df["datetime"].min(), "->", df["datetime"].max())
