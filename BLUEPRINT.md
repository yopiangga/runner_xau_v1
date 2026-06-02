# BLUEPRINT — Bot Sinyal AI Trading XAUUSD

Dokumen ini menjelaskan **arsitektur, keputusan desain, dan cara membangun ulang**
sistem sinyal AI untuk XAUUSD (gold), supaya Anda bisa membuat versi baru dengan
penyesuaian (simbol lain, timeframe lain, strategi lain, model lain).

---

## 1. TUJUAN SISTEM

Menghasilkan **sinyal BUY/SELL otomatis** pada XAUUSD yang:
- **TP tetap** (jarak konstan dari entry), **SL variatif** (mengikuti support/resistance).
- **Winrate tinggi** — dicapai dengan menyaring sinyal lewat ambang probabilitas model.
- Dieksekusi **manual oleh manusia**, dengan notifikasi tiap candle tutup.

Filosofi inti: **model = penyaring kualitas, bukan peramal harga.** Model menjawab
satu pertanyaan biner: *"Jika saya entry sekarang, apakah TP akan kena lebih dulu
daripada SL?"* Hanya sinyal berprobabilitas tinggi yang dieksekusi.

---

## 2. ARSITEKTUR & ALUR DATA

```
                 ┌─────────────┐
   AllTick API ─▶│ data fetcher │─▶ cache CSV (OHLCV)
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  features   │  indikator teknikal + deteksi support/resistance
                 └─────────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │  labeling   │  triple-barrier: label 1 (TP dulu) / 0 (SL dulu)
                 └─────────────┘
                        │
                        ▼
            ┌───────────────────────┐
            │ model (gradient boost)│  walk-forward train/test + threshold
            └───────────────────────┘
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
   backtest ideal   backtest      live runner
   (winrate OOS)    realistis     (notif tiap candle)
                    (+biaya)
```

---

## 3. KOMPONEN (modul demi modul)

| File | Peran | Hal penting |
|---|---|---|
| `config.py` | Semua parameter terpusat | Loader `.env` manual (tanpa dependency). Ubah strategi di sini. |
| `alltick_client.py` | Ambil data dari AllTick | Paginasi mundur, anti rate-limit, cache CSV, `update_cache()` inkremental |
| `features.py` | Feature engineering | Indikator murni pandas/numpy + swing low/high → support/resistance |
| `labeling.py` | Pelabelan target | Triple-barrier, parameter `side` (long/short) |
| `model.py` | Latih & backtest | `walk_forward()` generik (bisa model apa pun), `evaluate()`, `threshold_sweep()` |
| `run_experiment.py` | Eksperimen 1-arah (BUY) | Pipeline lengkap + laporan |
| `run_experiment_2way.py` | Eksperimen 2-arah (BUY+SELL) | Gabung long+short jadi 1 dataset dengan fitur `side` |
| `realistic_backtest.py` | Backtest realistis | Entry next-open, spread, slippage, 1-posisi, cooldown |
| `compare_timeframes.py` | Uji 1h vs 5m | Pilih timeframe optimal |
| `compare_algos.py` | Uji 10 algoritma ML | Pilih model terbaik (adil via top-K%) |
| `live_runner.py` | Runner otomatis | Loop tiap candle, notif macOS, retrain berkala |
| `live_signal.py` | Sinyal sekali jalan | Kartu rekomendasi candle terakhir |
| `evaluate_log.py` | Evaluasi sinyal nyata | Cek log live mana yang benar/salah |
| `analyze_hours.py` | Analisis jam sinyal | Kapan paling sering sinyal keluar (WIB) |

---

## 4. KEPUTUSAN DESAIN PENTING (& alasannya)

### 4.1 Sumber data: AllTick
- Provider diambil dari `.env` (`DATA_PROVIDER`, `ALLTICK_TOKEN`).
- **Kode simbol gold = `GOLD`** (bukan "XAUUSD" — itu `604 unauthorized`).
- Endpoint kline: `https://quote.alltick.io/quote-b-api/kline`.
- `kline_type`: 1=1m, 2=5m, 3=15m, 4=30m, 5=1h, 8=1d.
- **Rate limit ketat** (~1 req/detik free tier) → wajib backoff + cache.
- ~750–1000 bar per request → paginasi via `kline_timestamp_end`.

### 4.2 Strategi: TP tetap, SL variatif
- **TP tetap** = `entry ± TP_POINTS` (konstan). Mudah dieksekusi, target jelas.
- **SL variatif** = di support terdekat (BUY) / resistance terdekat (SELL).
  - Support = swing low tertinggi yang masih di bawah harga (dalam `SUPPORT_LOOKBACK` bar).
  - Resistance = swing high terendah di atas harga.
  - SL = level ± `SL_BUFFER_ATR * ATR` (buffer agar tak kena wick).
- Setup difilter: hanya valid jika `MIN_SL_POINTS ≤ jarak SL ≤ MAX_SL_POINTS`.

### 4.3 Pelabelan: triple-barrier
Untuk tiap bar (entry = close), pindai ke depan maksimal `HOLD_HORIZON` bar:
- BUY: label **1** jika `high ≥ TP` lebih dulu, **0** jika `low ≤ SL` lebih dulu.
- SELL: kebalikannya.
- Ambigu (TP & SL kena di 1 bar) → **konservatif: anggap SL dulu**.
- Tak ada yang kena dalam horizon → `NaN` (dibuang / TIME).

### 4.4 Model: Gradient Boosting + walk-forward
- **HistGradientBoostingClassifier** menang di perbandingan 10 algoritma (AUC 0.643).
- **Walk-forward** (expanding window): latih masa lalu, uji masa depan → tidak ada
  kebocoran data (look-ahead bias).
- Output = **probabilitas** P(TP dulu). Eksekusi hanya jika `proba ≥ PROB_THRESHOLD`.
- **Threshold = tuas winrate vs jumlah trade.** Makin tinggi → sedikit tapi akurat.

### 4.5 Dua arah (BUY + SELL)
- Bangun label long & short terpisah, gabung jadi 1 dataset, tambah fitur `side` (0/1).
- Satu model menangani keduanya → peluang ~2x, AUC malah naik.

### 4.6 Realisme untuk trading manual
Backtest ideal terlalu optimis. Versi realistis menambahkan:
1. Entry di **open candle berikutnya** (manusia bereaksi setelah candle tutup).
2. **Spread** (dibayar saat masuk).
3. **Slippage** entry & saat kena SL.
4. **Satu posisi** dalam satu waktu (bukan puluhan tumpang tindih).
5. **Cooldown** antar trade.

---

## 5. PARAMETER UTAMA (`config.py`)

```python
SYMBOL = "GOLD"             # kode AllTick untuk XAUUSD
KLINE_TYPE = 2              # 2=5m, 5=1h, dst
TOTAL_BARS = 14000          # jumlah bar historis untuk training

TP_POINTS = 2.0             # TP tetap (USD)
SL_BUFFER_ATR = 0.25        # buffer SL di luar support/resistance
MIN_SL_POINTS = 0.6         # batas bawah jarak SL
MAX_SL_POINTS = 4.0         # batas atas jarak SL (skip jika support terlalu jauh)
SWING_WINDOW = 5            # window deteksi swing low/high
SUPPORT_LOOKBACK = 60       # cari support/resistance dalam N bar
HOLD_HORIZON = 48           # horizon trade maksimum (bar)

PROB_THRESHOLD = 0.78       # ambang eksekusi (TUAS winrate)
N_SPLITS = 5                # fold walk-forward
RETRAIN_HOURS = 12          # interval retrain live runner
```

---

## 6. TEMUAN & PELAJARAN (penting untuk versi baru)

1. **RR < 1 itu rawan.** TP $2 vs SL rata-rata ~$3 → winrate impas ≈ 59%.
   Di bawah ambang ~0.70 winrate jatuh < 59% → **RUGI**. Winrate tinggi itu wajib,
   bukan bonus. *Jika ingin RR > 1, perbesar TP atau perketat MAX_SL_POINTS.*
2. **Threshold adalah kendali utama.** 0.60→57% (rugi), 0.70→75%, 0.78→86%, 0.80→93%.
3. **Timeframe**: 5m memberi winrate & frekuensi lebih tinggi dari 1h, TAPI TP harus
   dikecilkan (2–3, bukan 6) dan lebih rentan biaya/spread.
4. **Bias rezim pasar**: data uji hanya ~2,5 bulan (tren turun) → SELL dominan.
   Winrate tinggi **belum tentu bertahan** di pasar sideways/uptrend. Validasi lebih panjang.
5. **Jam terbaik (WIB)**: sesi London & NY (17:00–03:00 WIB) winrate 93–100%;
   sesi Asia (05–14 WIB) banyak sinyal tapi winrate rendah (77%).

---

## 7. CARA MEMBANGUN ULANG DENGAN PENYESUAIAN

### Ganti simbol (mis. EURUSD, BTCUSD)
- Ubah `SYMBOL` di `config.py` ke kode AllTick yang valid (uji dulu kode mana yang
  `ret=200`, lihat cara probing di awal `alltick_client.py`).
- Sesuaikan `TP_POINTS`, `MIN/MAX_SL_POINTS` ke skala harga aset itu (gold beda dgn forex).

### Ganti timeframe
- Ubah `KLINE_TYPE` (2=5m, 3=15m, 5=1h). Sesuaikan `HOLD_HORIZON` (mis. 5m butuh
  horizon lebih banyak bar untuk jarak yang sama) dan `TP_POINTS`.
- Jalankan `compare_timeframes.py` untuk bandingkan.

### Ganti strategi TP/SL
- TP dinamis (mis. kelipatan ATR): ubah `tp_price` di `labeling.py` & runner.
- SL pakai metode lain (ATR murni, fixed): ganti perhitungan `sl_price` di `labeling.py`.
- Target RR > 1: naikkan `TP_POINTS` atau turunkan `MAX_SL_POINTS`.

### Tambah/ubah fitur
- Tambah indikator di `features.py` `build_features()`, lalu daftarkan namanya di
  `FEATURE_COLS`. Model otomatis memakainya.

### Ganti model
- Ubah `_new_model()` di `model.py`, atau pakai `model_factory` di `walk_forward()`.
- Jalankan `compare_algos.py` untuk cari yang terbaik untuk data Anda.

### Ganti channel notifikasi
- Edit `notify()` di `live_runner.py`. Tambah Telegram (requests ke Bot API),
  email, atau webhook. Saat ini: bunyi + banner macOS + popup dialog.

---

## 8. URUTAN MENJALANKAN

```bash
pip install -r requirements.txt

# 1. Tarik data & eksperimen dasar
python3 run_experiment.py            # 1-arah (BUY)
python3 run_experiment_2way.py       # 2-arah (BUY+SELL)

# 2. Riset/optimasi
python3 compare_timeframes.py        # pilih timeframe
python3 compare_algos.py             # pilih algoritma
python3 realistic_backtest.py 0.78   # backtest realistis @ threshold

# 3. Live
python3 live_signal.py --fetch       # cek sinyal sekali
python3 live_runner.py               # jalan terus + notif
nohup python3 live_runner.py > runner.log 2>&1 &   # background

# 4. Evaluasi hasil nyata
python3 evaluate_log.py --conf 78    # mana sinyal yang benar/salah
python3 analyze_hours.py             # jam paling ramai sinyal
```

---

## 9. KETERBATASAN & RISIKO (wajib diingat)

- **Bukan saran finansial.** Eksperimen riset.
- Backtest realistis pakai asumsi biaya tetap; **saat news spread melebar liar**.
- Data historis 5m terbatas (~2,5 bulan) — overfit & bias rezim nyata.
- RR < 1 → disiplin SL mutlak; sekali SL digeser, matematika hancur.
- Free tier AllTick rate-limited → jangan fetch agresif.
- **Wajib paper-trade** beberapa minggu sebelum uang sungguhan.

---

## 10. DEPENDENCY

```
pandas, numpy, scikit-learn, joblib, requests
```
Python 3.9+. Notifikasi macOS pakai `osascript`/`afplay` bawaan (tanpa instalasi).
```
```
