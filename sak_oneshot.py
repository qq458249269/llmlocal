#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_oneshot.py — one-shot 能否把格式合规率救回来

思路：0.5B 记不住「只用 <tool_call> 包 JSON」这条规则，但照抄示例很行。
示例放在 history 最前面 = 常量前缀 = 还能 100% 命中前缀缓存。
    python sak_oneshot.py <port> <alias> <temp> [轮数]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]); MODEL = sys.argv[2]; TEMP = float(sys.argv[3])
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
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

# ---- one-shot 示例：用真实的 OpenAI 结构，chat template 会渲染成 <tool_call> XML ----
SHOT = [
    {"role": "user", "content": "看看 src 目录里有哪些文件"},
    {"role": "assistant", "tool_calls": [{"id": "call_demo", "type": "function", "function": {
        "name": "list_dir", "arguments": json.dumps({"path": "src"}, ensure_ascii=False)}}]},
    {"role": "tool", "tool_call_id": "call_demo", "name": "list_dir",
     "content": "main.py  utils.py"},
    {"role": "assistant", "content": "src 目录里有 main.py 和 utils.py 两个文件。"},
]

QS = [("看看 models 目录里有哪些文件", "list_dir"),
      ("读一下 README.md 的开头", "read_file"),
      ("在 notes/demo.txt 写入三行：本地 agent 循环测试", "write_file"),
      ("用两句话解释什么是前缀缓存", None)]


def post(msgs, maxtok=400):
    body = {"model": MODEL, "messages": msgs, "max_tokens": maxtok,
            "temperature": TEMP, "top_k": 20, "top_p": 0.8, "seed": 42,
            "tools": TOOLS, "tool_choice": "auto",
            "stop": ["<|im_end|>", "\nuser", "\nsystem"]}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r), time.time() - t0


print("model=%s temp=%.2f  one-shot=%d 条示例\n" % (MODEL, TEMP, 2))
ok_all = 0
for q, want in QS:
    msgs = [{"role": "system", "content": SYS}] + SHOT + [{"role": "user", "content": q}]
    d, wall = post(msgs)
    ch = d["choices"][0]
    m = ch["message"]
    tcs = m.get("tool_calls") or []
    names = [t["function"]["name"] for t in tcs]
    ok = (names[0] == want) if want else (not tcs and bool((m.get("content") or "").strip()))
    ok_all += ok
    print("[%s] %s" % ("PASS" if ok else "FAIL", q))
    print("   calls=%s finish=%s %d tok %.1fs" %
          (names or "无", ch.get("finish_reason"), d["usage"]["completion_tokens"], wall))
    for t in tcs:
        print("   args=%s" % t["function"]["arguments"][:140])
    if not tcs:
        print("   文本=%r" % (m.get("content") or "")[:200])
    print()
print("one-shot 通过 %d/%d" % (ok_all, len(QS)))