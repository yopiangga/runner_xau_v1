#!/bin/bash
# Jalankan live_runner terus-menerus:
# - caffeinate -i  : cegah Mac tidur (idle sleep) selama runner jalan
# - loop while     : restart otomatis kalau runner crash/keluar
# - /usr/bin/python3: interpreter yang punya pandas/sklearn (BUKAN homebrew)
#
# Pakai:  bash keepalive.sh
# Stop :  Ctrl+C  (atau: pkill -f live_runner.py)

cd "/Users/yopiangga/Documents/Riset/64 Bot Forex/4" || exit 1
PY=/usr/bin/python3

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] >>> start live_runner.py"
  caffeinate -i "$PY" live_runner.py
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] !!! runner berhenti, restart dalam 10 detik..."
  sleep 10
done
