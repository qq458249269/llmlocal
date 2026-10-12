#!/bin/bash
# ctxtune.sh — 在「真实长上下文」下调 KV 量化 / 线程 / fa / mlock
# 用法: bash ctxtune.sh "<长度列表>" "<变体1>" "<变体2>" ...
set -u
cd "$(dirname "$0")"
LENS="${1:-8192}"; shift
BASE="-ngl 0 -c 16384 -n 0 --parallel 1 --cache-ram 2048 --ctx-checkpoints 32 \
--checkpoint-min-step 256 --cache-reuse 64 --cache-idle-slots --context-shift \
--jinja -rea off --temp 0.1 --top-k 50"
i=0
for EX in "$@"; do
  i=$((i+1))
  LOG="/tmp/ct_$i.log"
  ./Bonsai-demo/bin/cpu/llama-server.exe -m models/LFM2.5-2.6B-Q4_K_M.gguf \
--alias p $BASE $EX --host 127.0.0.1 --port 8099 > "$LOG" 2>&1 &
  for k in $(seq 1 200); do curl -s --max-time 2 http://127.0.0.1:8099/health | grep -q ok && break; sleep 1; done
  if ! curl -s --max-time 2 http://127.0.0.1:8099/health | grep -q ok; then
    echo "  [ERR] server 没起来: $(tail -1 "$LOG")"; exit 1
  fi
  sleep 1
  echo "######## [$EX]"
  PYTHONIOENCODING=utf-8 python bench_ctx2.py 8099 p "$LENS" 2>&1 | sed 's/^/  /'
  powershell -NoProfile -Command "Get-Process llama-server -EA 0 | Stop-Process -Force" 2>/dev/null
  sleep 2
done
exit 0