#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_forced.py — 试 llama.cpp 的「强制工具」路径：tool_choice 指定函数对象

如果 llama.cpp 走 grammar 约束，输出必须已经是合法 tool_call，
不需要模型自己记得吐 <tool_call> 标签。
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

TOOLS = [{"type": "function", "function": {
    "name": "list_dir", "description": "列出仓库内某个目录下的文件名",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}}]

SYS = "你是一个本地命令行 agent，用中文回答。需要看目录时必须调用 list_dir 工具。"

for label, tc in (("tool_choice=required", "required"),
                  ("tool_choice=list_dir", {"type": "function", "function": {"name": "list_dir"}})):
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYS},
                                         {"role": "user", "content": "看看 models 目录里有哪些文件"}],
            "max_tokens": 120, "temperature": 0.1, "top_k": 20, "seed": 42,
            "tools": TOOLS, "tool_choice": tc}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=300))
    m = d["choices"][0]["message"]
    print("[%s] finish=%s  %.1fs" % (label, d["choices"][0]["finish_reason"], time.time() - t0))
    print("   tool_calls=%s" % json.dumps(m.get("tool_calls"), ensure_ascii=False))
    print("   content  =%r" % (m.get("content") or "")[:200])
    print()