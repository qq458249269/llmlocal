#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
raw_probe.py — 看模型吐的**原始文本**（不过 llama.cpp 的 tool_call 解析器）

    python raw_probe.py <port> <alias> <temp> [top_k]
"""
import json
import sys
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
TEMP = float(sys.argv[3]); TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 20
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "查询某个城市当前天气",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名"},
        "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}]

QS = ["上海现在天气怎么样？用摄氏度",
      "Read D:/AI/LLMlocal/models and tell me how many model files there are."]


def post(msgs, maxtok=300):
    body = {"model": MODEL, "messages": msgs, "max_tokens": maxtok,
            "temperature": TEMP, "top_k": TOPK, "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


for q in QS:
    d = post([{"role": "user", "content": q}])
    m = d["choices"][0]["message"]
    print("Q: %s   temp=%.2f" % (q, TEMP))
    print("   tool_calls : %s" % (json.dumps(m.get("tool_calls"), ensure_ascii=False) or "无"))
    print("   raw content: %r" % m.get("content"))
    print()