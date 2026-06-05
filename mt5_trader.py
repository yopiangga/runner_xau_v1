"""Eksekusi order OTOMATIS via MetaTrader5 (KHUSUS Windows).

Membuka posisi market (BUY/SELL) sesuai sinyal model dari live_runner, lengkap
dengan Stop Loss & Take Profit. Memakai koneksi terminal MT5 yang SAMA dengan
mt5_client (satu inisialisasi dipakai bersama untuk data + eksekusi).

Prasyarat (Windows):
  1. pip install MetaTrader5
  2. Terminal MetaTrader 5 ter-install & login ke akun broker (DEMO dulu!).
  3. Set di .env:
       AUTO_TRADE=1               # aktifkan auto-trading
       MT5_SYMBOL=XAUUSD          # sesuaikan nama simbol di broker
       TRADE_LOT=0.01             # ukuran lot per posisi
       MAX_OPEN_POSITIONS=1       # batas posisi bot yang terbuka bersamaan
       (login otomatis opsional via MT5_LOGIN/MT5_PASSWORD/MT5_SERVER)

PERINGATAN: order ini memakai dana sungguhan jika akun live. Selalu uji di akun
DEMO terlebih dulu. Di macOS/Linux modul ini tidak aktif (is_available()=False)
dan live_runner otomatis turun ke mode notifikasi saja.
"""
import platform
import config

try:
    import MetaTrader5 as mt5
except ImportError:  # bukan Windows / paket belum diinstall
    mt5 = None

# Pakai ulang init koneksi & util simbol dari klien data (satu sesi MT5).
import mt5_client


def is_available():
    """True hanya bila berjalan di Windows dengan paket MetaTrader5 terpasang."""
    return platform.system() == "Windows" and mt5 is not None


def _pick_filling(sym):
    """Tentukan mode filling yang didukung simbol (FOK/IOC/RETURN)."""
    mapping = {
        "ioc": mt5.ORDER_FILLING_IOC,
        "fok": mt5.ORDER_FILLING_FOK,
        "return": mt5.ORDER_FILLING_RETURN,
    }
    if config.TRADE_FILLING in mapping:
        return mapping[config.TRADE_FILLING]
    # auto: baca bitmask filling_mode simbol (2=IOC, 1=FOK)
    info = mt5.symbol_info(sym)
    fm = getattr(info, "filling_mode", 0) if info else 0
    if fm & 2:
        return mt5.ORDER_FILLING_IOC
    if fm & 1:
        return mt5.ORDER_FILLING_FOK
    return mt5.ORDER_FILLING_RETURN


def count_open_positions(sym=None):
    """Jumlah posisi bot (magic = config.TRADE_MAGIC) yang sedang terbuka."""
    if not is_available():
        return 0
    mt5_client._ensure_init()
    sym = sym or config.MT5_SYMBOL
    positions = mt5.positions_get(symbol=sym)
    if positions is None:
        return 0
    return sum(1 for p in positions if p.magic == config.TRADE_MAGIC)


def open_position(side, sl_price, tp_price, lot=None, comment="goldbot"):
    """Buka posisi market sesuai sinyal model.

    side: "BUY" atau "SELL". sl_price/tp_price: harga absolut SL & TP.
    Mengembalikan (result, error_str). Jika sukses error_str=None; result
    adalah objek OrderSendResult MT5 (punya .order, .price, .volume).
    """
    if not is_available():
        return None, "MT5 tidak tersedia (hanya Windows + paket MetaTrader5)."
    try:
        mt5_client._ensure_init()
        sym = mt5_client._symbol()
    except Exception as e:
        return None, f"init/simbol MT5 gagal: {e}"

    lot = float(lot or config.TRADE_LOT)
    tick = mt5.symbol_info_tick(sym)
    info = mt5.symbol_info(sym)
    if tick is None or info is None:
        return None, f"tidak dapat tick/info untuk {sym}."

    if side == "BUY":
        order_type, price = mt5.ORDER_TYPE_BUY, tick.ask
    elif side == "SELL":
        order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
    else:
        return None, f"side tidak dikenal: {side}"

    digits = info.digits
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": sym,
        "volume": lot,
        "type": order_type,
        "price": float(price),
        "sl": round(float(sl_price), digits),
        "tp": round(float(tp_price), digits),
        "deviation": config.TRADE_DEVIATION,
        "magic": config.TRADE_MAGIC,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": _pick_filling(sym),
    }
    result = mt5.order_send(request)
    if result is None:
        return None, f"order_send None: {mt5.last_error()}"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return result, f"order ditolak retcode={result.retcode} ({result.comment})"
    return result, None


if __name__ == "__main__":
    # Uji ringan: cek ketersediaan & posisi terbuka (tidak membuka order).
    print("MT5 tersedia:", is_available())
    if is_available():
        print("posisi bot terbuka:", count_open_positions())
