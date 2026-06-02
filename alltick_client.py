"""Klien AllTick: ambil data historis XAUUSD (kode 'GOLD') dengan paginasi,
penanganan rate-limit, dan cache ke CSV."""
import os, json, time, uuid, warnings
import requests
import pandas as pd
import config

warnings.filterwarnings("ignore")


def _request_kline(code, kline_type, num, end_ts, retries=6):
    """Satu panggilan kline. Menangani 429 (rate limit) dengan backoff."""
    query = {
        "trace": str(uuid.uuid4()),
        "data": {
            "code": code,
            "kline_type": kline_type,
            "kline_timestamp_end": end_ts,
            "query_kline_num": num,
            "adjust_type": 0,
        },
    }
    url = f"{config.ALLTICK_HOST}/kline"
    params = {"token": config.ALLTICK_TOKEN, "query": json.dumps(query)}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
        except Exception as e:
            print(f"  [warn] request error: {e}; retry...")
            time.sleep(5)
            continue
        if r.status_code == 429 or "Too many" in r.text:
            time.sleep(8)
            continue
        j = r.json()
        if j.get("ret") != 200:
            raise RuntimeError(f"AllTick error ret={j.get('ret')} msg={j.get('msg')}")
        return j["data"]["kline_list"]
    raise RuntimeError("Gagal: terus-menerus kena rate limit.")


def _bars_to_df(bars):
    rows = []
    for b in bars:
        ts = int(b["timestamp"])
        rows.append({
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="s"),
            "open": float(b["open_price"]),
            "high": float(b["high_price"]),
            "low": float(b["low_price"]),
            "close": float(b["close_price"]),
            "volume": float(b.get("volume", 0) or 0),
        })
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def fetch_recent(code=None, kline_type=None, num=400):
    """Fetch RINGAN: 1 request bar terbaru saja (untuk pemakaian live).
    Tidak menulis cache, tidak paginasi."""
    code = code or config.SYMBOL
    kline_type = kline_type or config.KLINE_TYPE
    bars = _request_kline(code, kline_type, num=num, end_ts=0)
    return _bars_to_df(bars)


def update_cache(code=None, kline_type=None, cache_path=None, recent_num=1000):
    """Opsi B: sambung candle BARU ke cache lalu kembalikan data ter-update.
    Hemat request (1 panggilan), cache selalu fresh untuk retraining live."""
    code = code or config.SYMBOL
    kline_type = kline_type or config.KLINE_TYPE
    cache_path = cache_path or config.CACHE_PATH

    if not os.path.exists(cache_path):
        # belum ada cache -> tarik penuh sekali
        return fetch_klines(code, kline_type, cache_path=cache_path)

    old = pd.read_csv(cache_path, parse_dates=["datetime"])
    new = fetch_recent(code, kline_type, num=recent_num)

    last_old = int(old["timestamp"].max())
    oldest_new = int(new["timestamp"].min())
    if oldest_new > last_old + 1:
        print(f"  [warn] ada celah data: cache s/d {last_old}, data baru mulai "
              f"{oldest_new}. Pertimbangkan force fetch penuh.")

    merged = pd.concat([old, new], ignore_index=True)
    # buang duplikat timestamp, simpan nilai TERBARU (candle terakhir bisa berubah)
    merged = merged.drop_duplicates(subset="timestamp", keep="last")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    added = len(merged) - len(old)
    merged.to_csv(cache_path, index=False)
    print(f"[update] cache: {len(old)} -> {len(merged)} bar (+{added} baru), "
          f"terbaru {merged['datetime'].iloc[-1]}")
    return merged


def fetch_klines(code=None, kline_type=None, total_bars=None,
                 cache_path=None, force=False):
    """Ambil `total_bars` kline historis dengan paginasi mundur.
    Hasil di-cache ke CSV agar tidak memukul API berulang."""
    code = code or config.SYMBOL
    kline_type = kline_type or config.KLINE_TYPE
    total_bars = total_bars or config.TOTAL_BARS
    cache_path = cache_path or config.CACHE_PATH

    if cache_path and os.path.exists(cache_path) and not force:
        df = pd.read_csv(cache_path, parse_dates=["datetime"])
        print(f"[cache] dimuat {len(df)} bar dari {cache_path}")
        return df

    print(f"[fetch] mengambil ~{total_bars} bar {code} (kline_type={kline_type}) dari AllTick...")
    all_bars = {}
    end_ts = 0
    page = 0
    while len(all_bars) < total_bars:
        page += 1
        bars = _request_kline(code, kline_type, num=1000, end_ts=end_ts)
        if not bars:
            break
        for b in bars:
            all_bars[int(b["timestamp"])] = b
        oldest = min(int(b["timestamp"]) for b in bars)
        print(f"  page {page}: +{len(bars)} bar (total {len(all_bars)}), oldest={oldest}")
        if end_ts != 0 and oldest >= end_ts:
            break  # tidak ada data lebih lama
        end_ts = oldest - 1
        time.sleep(2)  # sopan terhadap rate limit
        if page > 30:
            break

    rows = []
    for ts, b in sorted(all_bars.items()):
        rows.append({
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="s"),
            "open": float(b["open_price"]),
            "high": float(b["high_price"]),
            "low": float(b["low_price"]),
            "close": float(b["close_price"]),
            "volume": float(b.get("volume", 0) or 0),
        })
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path, index=False)
        print(f"[cache] disimpan {len(df)} bar ke {cache_path}")
    return df


if __name__ == "__main__":
    df = fetch_klines()
    print(df.tail())
    print("Rentang:", df["datetime"].min(), "->", df["datetime"].max())
