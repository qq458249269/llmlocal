#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ptc_probe.py -- does parallel_tool_calls:true fix LFM2.5 parallel tool calls?

The failure: model emits <|tool_call_start|>[fd(path='..'), rg(path='..')]<|tool_call_end|>
llama.cpp's lfm2_5 PEG parser only allows the ", second_call" tail when
parallel_tool_calls is on (chat-peg-parser.cpp:583).  pi never sends the field,
so server_params.parallel_tool_calls stays false, the parse fails, and the whole
turn degrades to plain content -> pi sees stopReason=stop and the agent exits.

Run with PTC=0 and PTC=1, same seed, compare finish_reason.
"""
import json, os, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
PTC  = os.environ.get("PTC", "omit")   # "omit" | "0" | "1"
URL  = "http://127.0.0.1:%d/v1/chat/completions" % PORT

TOOLS = [{"type": "function", "function": {"name": n, "description": d,
          "parameters": {"type": "object", "properties": p, "required": r}}}
         for n, d, p, r in [
    ("fd", "按路径/glob 查找文件, 永远优先于 find",
     {"path": {"type": "string", "description": "要搜索的目录"},
      "pattern": {"type": "string", "description": "glob 通配符"}}, ["path", "pattern"]),
    ("rg", "在文件中搜索内容, 永远优先于 grep",
     {"path": {"type": "string", "description": "要搜索的目录"},
      "pattern": {"type": "string", "description": "正则"}}, ["path", "pattern"]),
]]

body = {"model": "lfm25-2.6b", "stream": False, "seed": 7,
        "max_tokens": 900, "temperature": 0.1, "top_k": 50,
"messages": [{"role": "user", "content":
            "并行调用两个工具，别偷懒只调一个：fd(path='/AI/狗头军师', pattern='*') "
            "和 rg(path='/AI/狗头军师', pattern='pony|PONYTAIL')。两条都必须发。"}],
"tools": TOOLS, "tool_choice": "auto"}
# PTC=omit reproduces pi, which never sends the field -> server default decides.
if PTC != "omit":
    body["parallel_tool_calls"] = PTC == "1"

req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode(),
                             {"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(req, timeout=900).read().decode())
m = d["choices"][0]["message"]
print("parallel_tool_calls=%s  finish_reason=%s" % (PTC, d["choices"][0]["finish_reason"]))
ok = bool(m.get("tool_calls"))
if ok and PTC != "omit":
    args = [json.loads(c["function"]["arguments"]) for c in m["tool_calls"]]
    ok = len(args) == 2 and all("rg(" not in str(a) for a in args)
print("tool_calls :", json.dumps(m.get("tool_calls"), ensure_ascii=False))
print("content    :", repr((m.get("content") or "")[-300:]))
print("VERDICT    :", "OK  parsed" if ok else "FAIL mangled or degraded to text")
sys.exit(0 if ok else 1)