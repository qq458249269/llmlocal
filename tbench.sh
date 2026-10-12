#!/bin/bash
# tbench.sh — 固定 8k 上下文，比不同线程数 / 参数的 decode 速度
# 用法: bash tbench.sh <tag> "<额外参数>"
set -u
cd "$(dirname "$0")"
TAG="$1"; EX="$2"; LENS="${3:-8192}"
LOG="/tmp/tb_$TAG.log"; rm -f "$LOG"
./Bonsai-demo/bin/cpu/llama-server.exe \
  -m models/LFM2.5-8B-A1B-UD-Q4_K_M.gguf --alias probe -ngl 0 -fa on \
  -c "${CTX:-32768}" -ctk f16 -ctv f16 -n 0 --host 127.0.0.1 --port 8099 --parallel 1 \
  --context-shift --jinja -rea off --temp 0.2 --top-k 80 $EX > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 200); do
  curl -s --max-time 2 http://127.0.0.1:8099/health 2>/dev/null | grep -q ok && break
  sleep 1
done
sleep 2
echo "######## $TAG :: $EX"
PYTHONIOENCODING=utf-8 python bench_decode.py 8099 probe "$LENS" 2>&1 | sed 's/^/  /'
grep -E "eval time" "$LOG" | tail -1 | sed 's/^/  LOG /'
powershell -NoProfile -Command "Get-Process llama-server -EA 0 | ForEach-Object { '  >> PrivateWS {0:N2} GiB  WorkingSet {1:N2} GiB' -f (\$_.PrivateMemorySize64/1GB),(\$_.WorkingSet64/1GB) }" 2>/dev/null
powershell -NoProfile -Command "Stop-Process -Id $SRV -Force -EA 0" 2>/dev/null
sleep 2
exit 0
