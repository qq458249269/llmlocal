#!/usr/bin/env python
# TTFT / 速度测试:  测首字延迟与生成速度
# 用法:  python ttft_test.py [prompt_file] [max_tokens] [runs]
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "bonsai-1.7b"

path = sys.argv[1] if len(sys.argv) > 1 else "p17.txt"
maxt = int(sys.argv[2]) if len(sys.argv) > 2 else 150
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 2
doc = open(path, encoding="utf-8").read()
print("prompt: %s (%d chars)" % (path, len(doc)))

for run in range(1, runs + 1):
    body = json.dumps({
        "model": MODEL, "stream": True, "max_tokens": maxt, "temperature": 0.3,
        "messages": [{"role": "user", "content": doc}],
    }).encode()
    t0 = time.time(); first = None; n = 0; cached = None
    r = urllib.request.urlopen(urllib.request.Request(
        BASE, body, {"Content-Type": "application/json"}), timeout=600)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        if line == "data: [DONE]":
            break
        try:
            d = json.loads(line[6:])
        except ValueError:
            continue
        if d.get("usage"):
            cached = d["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        ch = (d.get("choices") or [{}])[0]
        if ch.get("delta", {}).get("content"):
            if first is None:
                first = time.time() - t0
            n += 1
    total = time.time() - t0
    print("run %d: TTFT %.2f s | gen %d tok @ %.1f t/s | total %.2f s | cached=%s"
          % (run, first or -1, n, n / max(total - (first or total), 0.001), total, cached))
