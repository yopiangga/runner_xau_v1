"""Broadcaster sinyal master-bot -> client-bot (socket, realtime).

TUJUAN: menambahkan kemampuan BROADCAST tanpa mengubah kode master-bot yang ada.
Modul ini berdiri sendiri (file BARU). Ia hanya MEMBACA `signals_log.csv` yang
sudah ditulis oleh live_runner.py, lalu mem-broadcast setiap sinyal BUY/SELL
(yang lolos threshold) ke semua client yang terhubung lewat TCP socket.

Cara pakai (jalankan BERDAMPINGAN dengan live_runner.py, di mesin yang sama):

  python3 live_runner.py            # master tetap jalan seperti biasa (TIDAK diubah)
  python3 signal_broadcaster.py     # broadcaster: sebar sinyal ke client via socket

Opsi:
  python3 signal_broadcaster.py --host 0.0.0.0 --port 9009
  python3 signal_broadcaster.py --demo      # kirim sinyal palsu tiap 10 dtk (uji client)
  python3 signal_broadcaster.py --replay     # broadcast juga sinyal lama di log (default: hanya baru)

Konfigurasi via .env (opsional, punya default):
  BROADCAST_HOST=0.0.0.0     # alamat bind server (0.0.0.0 = semua interface)
  BROADCAST_PORT=9009        # port server
  BROADCAST_LOG=signals_log.csv   # file log sinyal yang ditulis live_runner
  BROADCAST_SYMBOL=XAUUSD    # nama simbol yang dikirim ke client (hint saja)

PROTOKOL: newline-delimited JSON (NDJSON). Tiap pesan satu baris JSON diakhiri
'\\n'. Master = SERVER (publisher), client = subscriber. Master tidak membaca
balik dari client (komunikasi satu arah: broadcast).

Contoh pesan sinyal:
  {"type":"signal","id":"2026-06-23 14:35:00|BUY","symbol":"XAUUSD",
   "side":"BUY","price":3343.5,"tp":3345.5,"sl":3340.1,
   "tp_dist":2.0,"sl_dist":3.4,"confidence":0.86,
   "candle":"2026-06-23 14:35:00","ts":1750000000}
"""
import argparse
import csv
import json
import os
import socket
import sys
import threading
import time

# Pakai ulang loader .env & default simbol dari master (read-only, tidak mengubah master).
try:
    import config
    _DEFAULT_SYMBOL = config.MT5_SYMBOL
except Exception:
    config = None
    _DEFAULT_SYMBOL = "XAUUSD"


def _env(key, default):
    return os.environ.get(key, default)


HOST = _env("BROADCAST_HOST", "0.0.0.0")
PORT = int(_env("BROADCAST_PORT", "9009"))
LOG_FILE = _env("BROADCAST_LOG", "signals_log.csv")
SYMBOL = _env("BROADCAST_SYMBOL", _DEFAULT_SYMBOL)

POLL_SEC = 1.0            # interval cek file log (detik) -> latensi broadcast
HEARTBEAT_SEC = 15        # kirim heartbeat tiap N detik (deteksi koneksi mati)

# Kolom signals_log.csv (ditulis oleh live_runner.log_signal):
# waktu_cek, candle, close, sinyal, keyakinan, tp, sl, aksi
COLS = ["waktu_cek", "candle", "close", "sinyal", "keyakinan", "tp", "sl", "aksi"]


class Broadcaster:
    """TCP server pub-sub sederhana: terima banyak client, sebar pesan ke semua."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.clients = set()          # set socket client aktif
        self.lock = threading.Lock()
        self.server = None
        self.running = False
        self.sent_count = 0

    def start(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(16)
        self.running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        print(f"[broadcaster] server LISTEN {self.host}:{self.port} "
              f"(menunggu client...)")

    def _accept_loop(self):
        while self.running:
            try:
                conn, addr = self.server.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                self.clients.add(conn)
            print(f"[broadcaster] + client {addr[0]}:{addr[1]} "
                  f"(total {len(self.clients)})")
            # Salam pembuka supaya client tahu koneksi hidup.
            self._send_one(conn, {"type": "hello", "symbol": SYMBOL,
                                  "ts": int(time.time())})

    def _heartbeat_loop(self):
        while self.running:
            time.sleep(HEARTBEAT_SEC)
            self.broadcast({"type": "heartbeat", "ts": int(time.time())},
                           log=False)

    def _send_one(self, conn, msg):
        try:
            conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            return True
        except OSError:
            return False

    def broadcast(self, msg, log=True):
        """Kirim pesan ke SEMUA client. Client yang mati otomatis dibuang."""
        line = (json.dumps(msg) + "\n").encode("utf-8")
        dead = []
        with self.lock:
            targets = list(self.clients)
        for conn in targets:
            try:
                conn.sendall(line)
            except OSError:
                dead.append(conn)
        if dead:
            with self.lock:
                for conn in dead:
                    self.clients.discard(conn)
                    try:
                        conn.close()
                    except OSError:
                        pass
            print(f"[broadcaster] - {len(dead)} client terputus "
                  f"(sisa {len(self.clients)})")
        if log:
            self.sent_count += 1
            n = len(targets) - len(dead)
            print(f"[broadcaster] >> SINYAL {msg.get('side')} "
                  f"{msg.get('symbol')} @ {msg.get('price')} -> {n} client")

    def stop(self):
        self.running = False
        with self.lock:
            for conn in self.clients:
                try:
                    conn.close()
                except OSError:
                    pass
            self.clients.clear()
        if self.server:
            try:
                self.server.close()
            except OSError:
                pass


def row_to_signal(row):
    """Ubah satu baris log -> dict sinyal. Return None jika bukan BUY/SELL."""
    data = dict(zip(COLS, row))
    side = (data.get("aksi") or "").strip().upper()
    if side not in ("BUY", "SELL"):
        return None
    try:
        close = float(data["close"])
        tp = float(data["tp"])
        sl = float(data["sl"])
        conf = float(data["keyakinan"]) / 100.0
    except (ValueError, KeyError):
        return None
    candle = data.get("candle", "")
    # Jarak (distance) TP/SL relatif harga master. Client bisa hitung ulang
    # harga absolut dari tick broker-nya sendiri (lebih aman lintas-broker).
    if side == "BUY":
        tp_dist = round(tp - close, 2)
        sl_dist = round(close - sl, 2)
    else:  # SELL
        tp_dist = round(close - tp, 2)
        sl_dist = round(sl - close, 2)
    return {
        "type": "signal",
        "id": f"{candle}|{side}",
        "symbol": SYMBOL,
        "side": side,
        "price": round(close, 2),
        "tp": round(tp, 2),
        "sl": round(sl, 2),
        "tp_dist": tp_dist,
        "sl_dist": sl_dist,
        "confidence": round(conf, 4),
        "candle": candle,
        "waktu_cek": data.get("waktu_cek", ""),
        "ts": int(time.time()),
    }


def tail_log(bc, replay=False):
    """Ekor (tail) signals_log.csv: tiap baris BUY/SELL baru -> broadcast."""
    buffer = ""
    offset = 0
    started = False

    while bc.running:
        if not os.path.exists(LOG_FILE):
            if not started:
                print(f"[broadcaster] menunggu {LOG_FILE} dibuat oleh live_runner...")
                started = True
            time.sleep(POLL_SEC)
            continue

        if not started:
            # Default: hanya broadcast sinyal BARU setelah broadcaster start
            # (lewati histori). --replay untuk sebar histori juga.
            offset = 0 if replay else os.path.getsize(LOG_FILE)
            started = True
            print(f"[broadcaster] tail {LOG_FILE} "
                  f"({'replay histori + baru' if replay else 'hanya sinyal baru'})")

        try:
            size = os.path.getsize(LOG_FILE)
        except OSError:
            time.sleep(POLL_SEC)
            continue

        if size < offset:        # file ter-rotate / terpangkas -> reset
            offset, buffer = 0, ""
        if size > offset:
            with open(LOG_FILE, "r", newline="") as f:
                f.seek(offset)
                chunk = f.read()
                offset = f.tell()
            buffer += chunk
            *lines, buffer = buffer.split("\n")   # baris terakhir mungkin parsial
            for line in lines:
                if not line.strip() or line.startswith("waktu_cek"):
                    continue                       # lewati kosong / header
                try:
                    row = next(csv.reader([line]))
                except StopIteration:
                    continue
                sig = row_to_signal(row)
                if sig:
                    bc.broadcast(sig)

        time.sleep(POLL_SEC)


def demo_loop(bc):
    """Mode --demo: kirim sinyal palsu bergantian utk menguji client tanpa master."""
    print("[broadcaster] MODE DEMO: kirim sinyal palsu tiap 10 detik (Ctrl+C berhenti)")
    i = 0
    base = 3340.0
    while bc.running:
        time.sleep(10)
        i += 1
        side = "BUY" if i % 2 else "SELL"
        price = base + (i % 5)
        if side == "BUY":
            tp, sl = price + 2.0, price - 3.0
        else:
            tp, sl = price - 2.0, price + 3.0
        sig = {
            "type": "signal", "id": f"demo-{i}|{side}", "symbol": SYMBOL,
            "side": side, "price": round(price, 2),
            "tp": round(tp, 2), "sl": round(sl, 2),
            "tp_dist": 2.0, "sl_dist": 3.0, "confidence": 0.99,
            "candle": f"demo-{i}", "waktu_cek": "", "ts": int(time.time()),
        }
        bc.broadcast(sig)


def main():
    ap = argparse.ArgumentParser(description="Broadcaster sinyal master -> client (socket)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--demo", action="store_true", help="kirim sinyal palsu (uji client)")
    ap.add_argument("--replay", action="store_true",
                    help="broadcast juga sinyal lama yang sudah ada di log")
    args = ap.parse_args()

    bc = Broadcaster(args.host, args.port)
    try:
        bc.start()
    except OSError as e:
        print(f"[broadcaster] GAGAL bind {args.host}:{args.port}: {e}")
        sys.exit(1)

    print("=" * 60)
    print(f"  SIGNAL BROADCASTER (master-bot)  {SYMBOL}")
    print(f"  server  : {args.host}:{args.port}")
    print(f"  sumber  : {'DEMO (palsu)' if args.demo else LOG_FILE}")
    print("=" * 60)

    worker = demo_loop if args.demo else (lambda b: tail_log(b, replay=args.replay))
    t = threading.Thread(target=worker, args=(bc,), daemon=True)
    t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[broadcaster] dihentikan.")
    finally:
        bc.stop()


if __name__ == "__main__":
    main()
