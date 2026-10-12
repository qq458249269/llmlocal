#!/bin/bash
# sakstart.sh — 起 sakthai-0.5b 服务做测试（一次性验证用）
# 用法: bash sakstart.sh [端口] "<额外参数>"
set -u
cd "$(dirname "$0")"
PORT="${1:-8084}"; shift || true
EXTRA="${1:-}"
LOG=/tmp/sak_$PORT.log
./Bonsai-demo/bin/cpu/llama-server.exe \
  -m models/sakthai-0.5b-Q4_K_M.gguf --alias sakthai-0.5b \
  --host 127.0.0.1 --port "$PORT" --parallel 1 -ngl 0 -fa on \
  -c 16384 -ctk f16 -ctv f16 -t 4 -tb 4 -ub 512 -n 0 \
  --cache-ram 1024 --ctx-checkpoints 32 --checkpoint-min-step 256 \
  --cache-reuse 64 --cache-idle-slots --context-shift --jinja -rea off \
  --temp 0.7 --top-k 20 --top-p 0.8 --repeat-penalty 1.05 \
  $EXTRA > "$LOG" 2>&1 &
for k in $(seq 1 200); do curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break; sleep 1; done
sleep 1
if ! curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q ok; then
  echo "[ERR] server 没起来:"; tail -3 "$LOG"; exit 1
fi
echo "[ok] http://127.0.0.1:$PORT/v1  alias=sakthai-0.5b  log=$LOG"
exit 0