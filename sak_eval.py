#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_eval.py — sakthai-0.5b 的一次性评估：工具调用 + 中文分析 + 是否会停

对比不同 stop 序列 / 温度，顺便打印 finish_reason（length = 没停）。
    python sak_eval.py <port> <alias> <temp> [--stop]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TEMP = float(sys.argv[3])
USE_STOP = "--stop" in sys.argv
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
STOP = ["<|im_end|>", "user\n", "\nuser", "system\n", "<|endoftext|>"] if USE_STOP else None

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "列出仓库内某个目录下的文件名",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "读取仓库内一个文件的纯文本内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "把内容写入仓库内的一个文件（会覆盖）",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径"},
            "content": {"type": "string", "description": "文件内容"}},
            "required": ["path", "content"]}}},
]

SYS = ("你是一个本地命令行 agent，用中文回答。\n"
       "规则：\n1. 需要看目录、读文件或写文件时必须调用工具，不要凭空编造内容。\n"
       "2. 工具返回结果后，用中文简短总结，不要再调用工具。\n"
       "3. 只输出一次工具调用，然后停下来等结果。\n")

TASKS = [
    ("工具-选对", "看看 models 目录里有哪些文件", "list_dir"),
    ("工具-参数", "读一下 README.md 的开头", "read_file"),
    ("不该调", "用两句话解释什么是前缀缓存", None),
    ("中文分析", "读 kvcalc.py，然后用三句话中文说明它的作用", "read_file"),
]


def post(msgs, maxtok=400, tools=True):
    body = {"model": MODEL, "messages": msgs, "max_tokens": maxtok,
            "temperature": TEMP, "top_k": 20, "top_p": 0.8, "seed": 42}
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if STOP:
        body["stop"] = STOP
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r), time.time() - t0


print("model=%s temp=%.2f stop=%s\n" % (MODEL, TEMP, bool(STOP)))
bad = 0
for name, q, want in TASKS:
    d, wall = post([{"role": "system", "content": SYS}, {"role": "user", "content": q}])
    ch = d["choices"][0]
    m = ch["message"]
    tcs = m.get("tool_calls") or []
    names = [t["function"]["name"] for t in tcs]
    ok = bool(names and names[0] == want) if want else (not tcs)
    fin = ch.get("finish_reason")
    if fin == "length":
        ok = False
    bad += 0 if ok else 1
    print("[%s] %s  -> %s%s" % (name, q, "PASS" if ok else "FAIL",
                                "" if ok else "  <<<"))
    print("   tool_calls=%s  finish=%s  %d tok/%.1fs = %.1f t/s" %
          (names or "无", fin, d["usage"]["completion_tokens"], wall,
           d["usage"]["completion_tokens"] / max(wall, 1e-6)))
    for t in tcs:
        print("   args=%s = %s" % (t["function"]["name"], t["function"]["arguments"][:120]))
    if not tcs:
        print("   文本=%r" % (m.get("content") or "")[:220])
    print()
print("通过 %d/%d" % (len(TASKS) - bad, len(TASKS)))