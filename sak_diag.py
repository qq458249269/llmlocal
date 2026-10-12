#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_diag.py — SakThai 0.5B 行为诊断：把「循环说废话 / 返回内容是 JSON /
不吐 EOS」三件事量化成数字。

    python sak_diag.py <port> <alias> <temp> [每题采样数] [--stop]

每题跑 N 次，统计：
  eos     finish_reason 是不是 stop（=模型自己停了）；length = 说到上限被砍
  wrap    输出里有没有 <tool_call> 包装（llama.cpp 只认这个）
  json    输出正文本身是不是一个裸 JSON（含 name/arguments）
  loop    重复度 = 重复 12-gram 的比例，>0.3 基本就是卡在循环里
  附首 90 字，看它到底在说什么
"""
import json
import re
import sys
import urllib.request
from collections import Counter

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TEMP = float(sys.argv[3])
N = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 3
USE_STOP = "--stop" in sys.argv
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir", "description": "列出仓库内某个目录下的文件名",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "读取仓库内一个文件的纯文本内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}},
]

CASES = [
    ("工具", "看看 models 目录里有哪些文件", True),
    ("工具", "读一下 README.md 的开头", True),
    ("不用", "用两句话解释什么是前缀缓存", False),
    ("不用", "1+1 等于几？只回数字", False),
]


def post(msgs, maxtok=120, tools=True):
    body = {"model": MODEL, "messages": msgs, "max_tokens": maxtok,
            "temperature": TEMP, "top_k": 20, "top_p": 0.8}
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    if USE_STOP:
        body["stop"] = ["<|im_end|>", "\nuser", "\nassistant", "\nsystem", "\ntool\n"]
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def loop_ratio(t):
    w = re.findall(r"[A-Za-z一-鿿]+", t)
    if len(w) < 24:
        return 0.0
    g = ["".join(w[i:i + 12]) for i in range(len(w) - 11)]
    c = Counter(g)
    return 1.0 - len(c) / len(g)


print("model=%s temp=%.2f N=%d stop=%s" % (MODEL, TEMP, N, USE_STOP))
print("%-5s %-26s %-4s %-5s %-5s %-6s %-5s" %
      ("类型", "问题", "eos", "wrap", "json", "loop", "解析"))
agg = Counter()
for kind, q, want_tool in CASES:
    for i in range(N):
        d = post([{"role": "user", "content": q}])
        ch = d["choices"][0]
        m = ch["message"]
        txt = (m.get("content") or "").strip()
        has_tc = bool(m.get("tool_calls"))
        eos = ch.get("finish_reason") == "stop"
        wrap = "tool_call" in txt
        bare_json = bool(re.search(r'\{"name"\s*:', txt))
        lr = loop_ratio(txt)
        for k, v in (("eos", eos), ("wrap", wrap), ("json", bare_json),
                     ("loop", lr > 0.3), ("parsed", has_tc)):
            agg[k] += v
        print("%-5s %-26s %-4s %-5s %-5s %-6.2f %-5s | %s" %
              (kind, q[:24], "Y" if eos else "N", "Y" if wrap else "-",
               "Y" if bare_json else "-", lr, "OK" if has_tc else "--",
               txt[:90].replace("\n", "\\n")))
print()
tot = N * len(CASES)
print("汇总 %d/%d 样本：  自己停了 %d   带 tool_call 包装 %d   正文是裸 JSON %d   "
      "进循环 %d   被解析成 tool_calls %d" %
      (tot, tot, agg["eos"], agg["wrap"], agg["json"], agg["loop"], agg["parsed"]))