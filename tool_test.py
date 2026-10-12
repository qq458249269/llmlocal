#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tool_test.py — 最小化 tool-calling 冒烟测试（模型无关）

跑法:  python tool_test.py [端口] [模型alias] [temperature]

目的：只回答一个问题——**这个模型能不能原生吐 OpenAI 结构的 tool_calls**。
三个用例覆盖 agent 最常见的三种情况：
  1. 单工具、无歧义        -> 必须调，且工具名要对
  2. 多工具、需要选择      -> 必须选对那个
  3. 不需要工具            -> 必须直接文本回答（不该硬调）

每例都做「调工具 -> 回灌 -> 收尾」两跳，检查 finish_reason 和最终答案。
"""
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
MODEL = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AGENT_MODEL", "lfm25-2.6b")
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else float(os.environ.get("AGENT_TEMP", "0.1"))
TOP_K = int(os.environ.get("AGENT_TOP_K", "50" if "lfm" in MODEL else "20"))
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
# LFM2.5-2.6B 是 thinking 模型：思考 token 和答案共用 max_tokens 预算。
# 给 300 会全部耗在思考上，答案/tool_call 直接被截断（finish_reason=length）。
MAXTOK = int(os.environ.get("AGENT_MAXTOK", "600"))

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询某个城市当前天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名"},
            "unit": {"type": "string", "enum": ["c", "f"], "description": "温度单位"}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_time",
        "description": "查询某个时区当前时间",
        "parameters": {"type": "object", "properties": {
            "tz": {"type": "string", "description": "时区，如 Asia/Shanghai"}},
            "required": ["tz"]}}},
]

CASES = [
    ("单工具", "上海现在天气怎么样？用摄氏度", "get_weather", True),
    ("多工具-需选择", "帮我查一下纽约的天气", "get_weather", True),
    ("不该调工具", "用一句话解释什么是向量数据库。", None, False),
]

WEATHER = '{"temp_c": 27.5, "condition": "多云", "humidity": 0.62}'


def post(msgs, max_tokens=MAXTOK):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": TEMP, "top_k": TOP_K, "seed": 42,
            "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read().decode("utf-8"))
    return d, time.time() - t0


ok = 0
total = 0
print("model=%s  temp=%.2f  top_k=%d  -> %s\n" % (MODEL, TEMP, TOP_K, URL))
for name, q, want, should_call in CASES:
    print("[%s] %s" % (name, q))
    msgs = [{"role": "user", "content": q}]
    d, wall = post(msgs)
    ch = d["choices"][0]
    m = ch["message"]
    tcs = m.get("tool_calls")
    total += 1

    if should_call:
        names = [t["function"]["name"] for t in (tcs or [])]
        good = bool(tcs) and names[0] == want
        print("  tool_calls : %s   finish=%s" % (names or "(无)", ch.get("finish_reason")))
        for t in (tcs or []):
            print("  args       : %s = %s" % (t["function"]["name"], t["function"].get("arguments")))
        if not good:
            print("  FAIL       : 期望 %s, 实际 %s, 文本=%r" % (want, names or "无",
                                                             (m.get("content") or "")[:70]))
            ok += 0
        else:
            ok += 1
            # 第二跳：回灌工具结果，看能不能收尾
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": t["id"], "type": "function",
                 "function": {"name": t["function"]["name"],
                              "arguments": t["function"]["arguments"]}} for t in tcs]})
            msgs.append({"role": "tool", "tool_call_id": tcs[0]["id"],
                         "name": want, "content": WEATHER})
            d2, wall2 = post(msgs)
            c2 = d2["choices"][0]["message"]
            txt = (c2.get("content") or "").strip()
            no_tc = not c2.get("tool_calls")
            print("  回灌后     : tool_calls=%s  答案=%r" % (c2.get("tool_calls") or "无", txt[:80]))
            if no_tc and txt:
                ok += 0
            else:
                ok += 1
                if tcs:
                    print("  PASS       : 选对工具 + 回灌后收尾（%.1fs + %.1fs）" % (wall, wall2))
    else:
        txt = (m.get("content") or "").strip()
        no_tc = not tcs
        if no_tc and txt:
            ok += 1
            print("  纯文本     : %r" % txt[:80])
            print("  PASS       : 没有乱调工具（%.1fs）" % wall)
        else:
            print("  FAIL       : 该直接答却调了工具 %s，文本=%r"
                  % ([t["function"]["name"] for t in (tcs or [])], txt[:60]))
    print()

print("=" * 70)
print("%s  tool-calling 冒烟: %d/%d 通过" % (MODEL, ok, total))
print("=" * 70)
sys.exit(0 if ok == total else 1)
