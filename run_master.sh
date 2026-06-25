#!/bin/bash
# Jalankan live_runner.py + signal_broadcaster.py BERSAMAAN dalam satu perintah.
# - Keduanya WAJIB di mesin yang sama (broadcaster membaca signals_log.csv
#   yang ditulis live_runner).
# - Tiap proses di-restart otomatis bila crash/keluar.
# - Ctrl+C (atau kill) mematikan KEDUANYA dengan rapi.
# - Lintas-platform: pakai caffeinate (cegah Mac tidur) HANYA bila tersedia.
#
# Pakai:
#   bash run_master.sh                 # jalan di foreground (log juga ke file)
#   nohup bash run_master.sh >/dev/null 2>&1 &   # jalan di background (server)
#
# Stop:
#   Ctrl+C                       (jika foreground)
#   pkill -f run_master.sh       (jika background — hentikan SUPERVISOR, bukan
#                                 python-nya; mematikan python langsung percuma
#                                 karena akan di-restart oleh loop)
#   atau:  kill "$(cat run_master.pid)"
#
# Override interpreter:  PY=/usr/bin/python3 bash run_master.sh
set -u

# Pindah ke folder skrip ini (master-bot), apa pun direktori pemanggil.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1

PY="${PY:-/usr/bin/python3}"          # interpreter yg punya pandas/sklearn
PORT="${BROADCAST_PORT:-9009}"        # port broadcaster (boleh override via env)

# caffeinate hanya ada di macOS; di Linux dilewati.
if command -v caffeinate >/dev/null 2>&1; then
  PREFIX=(caffeinate -i)
else
  PREFIX=()
fi

run_forever() {            # $1=nama, $2=logfile, sisanya=argumen ke python
  local name="$1" logf="$2"; shift 2
  while true; do
    echo "[$(date '+%F %T')] >>> start $name" | tee -a "$logf"
    "${PREFIX[@]}" "$PY" "$@" >>"$logf" 2>&1
    echo "[$(date '+%F %T')] !!! $name berhenti, restart 10 detik..." | tee -a "$logf"
    sleep 10
  done
}

# Catat PID supervisor agar mudah dihentikan saat jalan di background.
echo $$ > run_master.pid

# Matikan SELURUH grup proses (kedua loop + anak python) saat Ctrl+C / kill.
trap 'echo; echo "[stop] mematikan live_runner + broadcaster..."; rm -f run_master.pid; kill 0' INT TERM

echo "============================================================"
echo "  MASTER SERVER: live_runner + signal_broadcaster"
echo "  python      : $PY"
echo "  broadcaster : port $PORT"
echo "  log         : live_runner.log , broadcaster.log"
echo "  stop        : Ctrl+C  |  pkill -f 'live_runner.py|signal_broadcaster.py'"
echo "============================================================"

# 1) Pencari sinyal (master, TIDAK diubah)
run_forever "live_runner" "live_runner.log" live_runner.py &

# 2) Broadcaster socket (sebar sinyal ke client-bot)
run_forever "broadcaster" "broadcaster.log" signal_broadcaster.py --port "$PORT" &

wait      # tahan skrip tetap hidup sampai di-Ctrl+C / di-kill
