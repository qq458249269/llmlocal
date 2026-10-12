#!/bin/bash
# ctxsweep.sh — 在「同一批模型」上测 decode vs 上下文长度，用于回答
#                「多少上下文才能达到 N t/s」
# 用法: bash ctxsweep.sh <model.gguf> <alias> "<server 参数>" [长度列表]
set -u
cd "$(dirname "$0")"
MODEL="$1"; ALIAS="$2"; SRVP="$3"; LENS="${4:-512,2048,8192,16384}"
LOG="/tmp/cs_$ALIAS.log"; rm -f "$LOG"
./Bonsai-demo/bin/cpu/llama-server.exe -m "$MODEL" --alias "$ALIAS" \
  --host 127.0.0.1 --port 8099 --parallel 1 --jinja -rea off -n 0 $SRVP > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 200); do
  curl -s --max-time 2 http://127.0.0.1:8099/health 2>/dev/null | grep -q ok && break
  sleep 1
done
sleep 1
echo "######## $ALIAS"
PYTHONIOENCODING=utf-8 python bench_decode.py 8099 "$ALIAS" "$LENS" 2>&1 | sed 's/^/  /'
powershell -NoProfile -Command "Get-Process llama-server -EA 0 | Stop-Process -Force -EA 0" 2>/dev/null
sleep 2
exit 0
