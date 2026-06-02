"""Konfigurasi eksperimen & loader .env (tanpa dependency tambahan)."""
import os
import platform

def load_env(path=".env"):
    """Parse file .env sederhana -> dict, juga inject ke os.environ."""
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                env[k] = v
                os.environ.setdefault(k, v)
    return env

ENV = load_env()

# --- Provider data ---
# "auto"   -> Windows pakai MT5, OS lain pakai AllTick (default)
# "alltick"-> paksa REST API AllTick (lintas-platform)
# "mt5"    -> paksa MetaTrader5 (hanya Windows)
_RAW_PROVIDER = ENV.get("DATA_PROVIDER", "auto").strip().lower()
if _RAW_PROVIDER in ("", "auto"):
    DATA_PROVIDER = "mt5" if platform.system() == "Windows" else "alltick"
else:
    DATA_PROVIDER = _RAW_PROVIDER
ALLTICK_TOKEN = ENV.get("ALLTICK_TOKEN", "")
ALLTICK_HOST = "https://quote.alltick.io/quote-b-api"
# Kode simbol XAUUSD (spot gold) di AllTick adalah "GOLD"
SYMBOL = "GOLD"

# --- Timeframe ---
# AllTick kline_type: 1=1m 2=5m 3=15m 4=30m 5=1h 6=2h 7=4h 8=1d
KLINE_TYPE = 2            # 5 menit (scalping)
TIMEFRAME_LABEL = "5m"
TOTAL_BARS = 14000        # ~ 2,5 bulan data 5 menit (via paginasi)
CACHE_PATH = f"data/{SYMBOL}_{TIMEFRAME_LABEL}.csv"

# --- MT5 (Windows: ambil data lewat terminal MetaTrader 5) ---
# Aktif jika DATA_PROVIDER=mt5. Sesuaikan MT5_SYMBOL dengan nama simbol broker.
MT5_SYMBOL = ENV.get("MT5_SYMBOL", "XAUUSD")
MT5_LOGIN = ENV.get("MT5_LOGIN", "")        # opsional (login otomatis)
MT5_PASSWORD = ENV.get("MT5_PASSWORD", "")  # opsional
MT5_SERVER = ENV.get("MT5_SERVER", "")      # opsional
MT5_PATH = ENV.get("MT5_PATH", "")          # opsional: path ke terminal64.exe
# Cache MT5 terpisah dari AllTick (zona waktu & nama simbol bisa berbeda)
MT5_CACHE_PATH = f"data/{MT5_SYMBOL}_{TIMEFRAME_LABEL}.csv"

# --- Strategi: TP TETAP (kecil utk scalping), SL VARIATIF (mengikuti support) ---
TP_POINTS = 2.0           # Take Profit tetap = 2.0 USD (≈ 200 pip gold)
SL_BUFFER_ATR = 0.25      # buffer di bawah support = 0.25 * ATR
MAX_SL_POINTS = 4.0       # SL maksimum (skip sinyal jika support terlalu jauh)
MIN_SL_POINTS = 0.6       # SL minimum (hindari noise terlalu rapat)
SWING_WINDOW = 5          # window deteksi swing low (support)
SUPPORT_LOOKBACK = 60     # cari support dalam 60 bar terakhir
HOLD_HORIZON = 48         # horizon maksimum trade = 48 bar (4 jam)

# --- Model ---
PROB_THRESHOLD = 0.75     # ambil sinyal hanya jika P(menang) >= threshold
                          # 0.78 = hanya sinyal kuat (winrate ~90%+), menguntungkan.
                          # CATATAN: di bawah ~0.70 winrate jatuh < breakeven (~59%)
                          # karena RR < 1 (TP $2 < SL rata2 ~$3) -> RUGI. Jangan turun.

# --- Notifikasi Telegram (kirim saat ada sinyal >= threshold) ---
TELEGRAM_BOT_TOKEN = ENV.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = ENV.get("TELEGRAM_CHAT_ID", "")

# --- Live runner ---
RETRAIN_HOURS = 12        # latih ulang model tiap N jam (data di-update dulu)
N_SPLITS = 5              # jumlah fold walk-forward
RANDOM_STATE = 42
