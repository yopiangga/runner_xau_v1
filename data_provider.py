"""Pemilih sumber data: AllTick (default, lintas-platform) atau MT5 (Windows).

Pilih lewat .env:  DATA_PROVIDER=alltick  |  DATA_PROVIDER=mt5

Semua modul lain mengimpor fetch_recent / update_cache / fetch_klines DARI SINI,
sehingga mengganti sumber data cukup mengubah satu variabel di .env tanpa
menyentuh logika model maupun runner. Kedua klien mengembalikan DataFrame
dengan skema yang sama: timestamp, datetime, open, high, low, close, volume.
"""
import config

_PROVIDER = None


def _provider():
    """Muat modul klien sesuai DATA_PROVIDER (lazy, hanya sekali)."""
    global _PROVIDER
    if _PROVIDER is None:
        name = (config.DATA_PROVIDER or "alltick").lower()
        if name in ("mt5", "metatrader5", "metatrader"):
            import mt5_client as p
        else:
            import alltick_client as p
        _PROVIDER = p
    return _PROVIDER


def fetch_recent(*args, **kwargs):
    return _provider().fetch_recent(*args, **kwargs)


def update_cache(*args, **kwargs):
    return _provider().update_cache(*args, **kwargs)


def fetch_klines(*args, **kwargs):
    return _provider().fetch_klines(*args, **kwargs)
