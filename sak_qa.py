#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_qa.py — 不给工具，纯中文问答/代码分析质量 + 会不会停
    python sak_qa.py <port> <alias> <temp>
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TEMP = float(sys.argv[3])
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

SYS = "你是一个中文命令行助手。回答用中文，不超过 5 行，不要重复自己的话，答完就停。"

QS = [
    "用一句话解释什么是前缀缓存（prompt cache）。",
    "下面这段 Python 在做什么？有什么问题？\n\n"
    "def f(d):\n    return [x for k, v in d.items() for x in v]",
    "我的 agent 每轮都要重算 2000 token 的系统提示，prefill 要 30 秒。"
    "llama.cpp 有哪个开关能让第二轮变快？简短说明。",
]


def post(msgs, maxtok=300):
    body = {"model": MODEL, "messages": msgs, "max_tokens": maxtok,
            "temperature": TEMP, "top_k": 20, "top_p": 0.8, "seed": 42,
            "stop": ["<|im_end|>", "\nuser", "\nsystem"]}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r), time.time() - t0


print("model=%s temp=%.2f (无工具, 带 stop)\n" % (MODEL, TEMP))
for q in QS:
    d, wall = post([{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    ch = d["choices"][0]
    txt = (ch["message"].get("content") or "").strip()
    print("Q: %s" % q[:60].replace("\n", " "))
    print("   finish=%s  %d tok  %.1fs  %.1f t/s" %
          (ch.get("finish_reason"), d["usage"]["completion_tokens"], wall,
           d["usage"]["completion_tokens"] / max(wall, 1e-6)))
    print("   A: %s" % txt[:400])
    print()