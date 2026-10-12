#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bench_ctx2.py — 用 server 端 timings 测 decode vs 真实 KV 长度

bench_decode.py 用「两次请求的墙钟差」反推 decode，这里直接读
llama.cpp 返回的 timings（prompt_ms / predicted_ms / predicted_per_second），
不受 client 计时噪声影响。

    python bench_ctx2.py <port> <alias> <长度列表>
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
MODEL = sys.argv[2] if len(sys.argv) > 2 else "probe"
LENS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                         else "512,2048,8192,16384".split(","))]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

FILLER = ("这是一段用于填充上下文的普通说明文字，描述本地推理环境、内存规划、"
          "缓存策略和工具调用流程。") * 4

SYSTEM = ("你是一个本地命令行 agent。规则：\n"
          "1. 需要外部信息或执行动作时，必须调用工具，不要凭空猜测。\n"
          "2. 回答用中文，简洁分点，不超过 5 行。\n") * 4


def post(msgs, max_tokens):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.1, "top_k": 50,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r), time.time() - t0


print("%-7s %-8s %-8s %-10s %-10s %-10s %-8s" %
      ("目标", "prompt_n", "cache_n", "prefill", "decode", "t/s", "wall"))
for target in LENS:
    n_filler = max(1, int(target / 130))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": FILLER * n_filler},
            {"role": "assistant", "content": "好的，我已读取背景。"},
            {"role": "user", "content": "用一句话总结上面的背景，然后停。"}]
    cold, _ = post(msgs, 1)                    # 冷：整段 prefill
    r, wall = post(msgs, 200)                      # 热：只算 LCP 之后的增量
    t = r["timings"]
    ct = cold["timings"]
    pf = ct["prompt_n"] / max(ct["prompt_ms"] / 1000.0, 1e-6)
    print("%-7d %-8d %-8d %-10.1f %-10.1f %-10.2f %-8.1f" %
          (target, t["prompt_n"] + t.get("cache_n", 0), t.get("cache_n", 0), pf,
           t["predicted_per_second"], t["predicted_per_second"], wall))