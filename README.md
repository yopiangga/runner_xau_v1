# Model AI — Prediksi & Aksi BUY XAUUSD (Spot Gold)

Eksperimen Python untuk memprediksi sinyal **BUY** pada XAUUSD dengan aturan:
- **Take Profit (TP) TETAP** — jarak konstan dari entry (default `6.0 USD`).
- **Stop Loss (SL) VARIATIF** — diletakkan tepat di bawah **support** (swing low)
  terdekat, jadi besarnya berbeda tiap sinyal.
- **Winrate tinggi** — model hanya mengeksekusi BUY bila probabilitas menang
  ≥ threshold, sehingga sedikit trade tapi berkualitas.

## Sumber Data
Diambil dari provider pada `.env` → **AllTick** (`DATA_PROVIDER=alltick`).
Simbol XAUUSD di AllTick memakai kode **`GOLD`**. Timeframe default **5 menit**
(scalping, `TP_POINTS=2.0`, `PROB_THRESHOLD=0.85`).

## Alur (pipeline)
1. `alltick_client.py` — ambil data historis (paginasi mundur, anti rate-limit, cache CSV).
2. `features.py` — indikator teknikal (EMA/RSI/MACD/ATR/ADX/Bollinger/Stoch + struktur candle) dan **deteksi support** (swing low) → jarak ke support.
3. `labeling.py` — pelabelan **triple-barrier**: label 1 bila TP tetap tersentuh lebih dulu, 0 bila SL (support) lebih dulu. Setup difilter agar SL dalam `[MIN_SL, MAX_SL]`.
4. `model.py` — `HistGradientBoostingClassifier` + **backtest walk-forward** (latih masa lalu, uji masa depan / out-of-sample) + sweep threshold.
5. `run_experiment.py` — menjalankan semuanya, melaporkan winrate, menyimpan model & sinyal terbaru.

## Cara Pakai
```bash
pip install -r requirements.txt
python3 run_experiment.py            # pakai cache bila ada
python3 run_experiment.py --fetch    # paksa ambil data baru dari AllTick
```

## Hasil Backtest (out-of-sample, 5m, TP=2.0, data ~2,5 bulan)
| Threshold P(menang) | Jumlah Trade | Winrate | Ekspektansi/trade |
|---|---|---|---|
| 0.70                 | 259 | 70.7% | +0.46 USD |
| 0.80                 | 88  | 79.5% | +0.92 USD |
| 0.85 (default)       | 34  | **91.2%** | +1.49 USD |
| 0.90                 | 6   | 83.3% (terlalu sedikit) | +1.04 USD |

Winrate tanpa model (ambil semua setup) = **54.7%**. Model menaikkannya hingga
**91%** pada threshold 0.85 dengan menyaring sinyal terbaik. Turunkan
`PROB_THRESHOLD` bila ingin lebih banyak trade (winrate sedikit turun).
Lihat `compare_timeframes.py` untuk perbandingan 1h vs 5m.

## Parameter penting (`config.py`)
- `TP_POINTS` — TP tetap (USD).
- `MIN_SL_POINTS` / `MAX_SL_POINTS` — batas jarak SL ke support.
- `PROB_THRESHOLD` — ambang probabilitas untuk eksekusi BUY.
- `KLINE_TYPE` — timeframe (5=1h, 3=15m, dst). 15m memberi lebih banyak sampel.

## Output
- `model_xauusd.joblib` — model final untuk prediksi live.
- `oos_trades.csv` — seluruh trade out-of-sample (untuk audit).
- `data/GOLD_1h.csv` — cache data mentah.

## Catatan & Risiko
- Ini eksperimen riset, **bukan saran finansial**. Backtest belum memperhitungkan
  spread/komisi/slippage — tambahkan sebelum live.
- RR rata-rata < 1 (TP 6 < SL rata-rata 7.4), jadi winrate tinggi memang
  diperlukan agar profitabel; ekspektansi tetap positif berkat filter model.
- Token AllTick free tier dibatasi rate (≈1 req/detik) — fetcher sudah menangani.
