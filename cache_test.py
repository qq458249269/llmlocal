#!/usr/bin/env python
# 会话缓存 / 前缀复用验证:  连发 N 次相同 system prompt + 不同问题
# 用法:  python cache_test.py [system_prompt_file] [rounds]
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8080/v1/chat/completions"
MODEL = "bonsai-1.7b"

path = sys.argv[1] if len(sys.argv) > 1 else "Bonsai-demo/README.md"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
sysmsg = open(path, encoding="utf-8").read()
print("system prompt: %s (%d chars)" % (path, len(sysmsg)))

questions = [
    "用一句话说明你能做什么。",
    "上面这份文档讲的是什么？一句话。",
    "文档里列了几个模型？只回答数字。",
]

for i in range(rounds):
    body = json.dumps({
        "model": MODEL, "stream": True, "max_tokens": 60, "temperature": 0.3,
        "stream_options": {"include_usage": True},
        "messages": [
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": questions[i % len(questions)]},
        ],
    }).encode()
    t0 = time.time(); first = None; n = 0; u = None
    r = urllib.request.urlopen(urllib.request.Request(
        BASE, body, {"Content-Type": "application/json"}), timeout=900)
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
            u = d["usage"]
        ch = (d.get("choices") or [{}])[0]
        if ch.get("delta", {}).get("content"):
            if first is None:
                first = time.time() - t0
            n += 1
    total = time.time() - t0
    pt = u.get("prompt_tokens", 0) if u else 0
    ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) if u else 0
    hit = (100.0 * ct / pt) if pt else 0
    print("轮 %d: TTFT %6.2f s | prompt %5d tok (复用 %5d = %5.1f%%) | 生成 %3d tok @ %5.1f t/s | 合计 %6.2f s"
          % (i + 1, first or -1, pt, ct, hit, n, n / max(total - (first or total), .001), total))
