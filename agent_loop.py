#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
agent_loop.py — 真实 agent 多轮工具循环测试（模型无关）

跑法:
    python agent_loop.py [轮数] [端口] [模型alias] [temperature]

例:
    python agent_loop.py 5 8080 lfm25-2.6b 0.1
    python agent_loop.py 5 8080 bonsai-1.7b  0.3

暖前缀（不跑循环，只把 system+tools 预填充一次后退出）:
    python agent_loop.py 0 8080 ornith-9b --warm
    用同一份 SYSTEM/TOOLS，所以字节级一致 —— 之后第一条真实请求直接命中
    前缀缓存。CPU 上 9B 的预填充约 15 t/s，4000 token = 4.5 min，把它挪到
    开机阶段就不用在提问时干等。

它测的是 agent 真正要用的四件事:
  1. tool_calls 能不能原生返回（finish_reason == "tool_calls"）
  2. 工具名 / 参数对不对（和该轮「期望调用的工具」比对）
  3. 工具结果回灌后能不能给出最终答案（不再空转调工具）
  4. 前缀缓存有没有真的命中（复用率）与生成速度

采样参数必须跟模型卡一致，否则工具调用的 JSON 会崩:
    LFM2.5-2.6B     temp 0.1  top_k 50
    Ternary-Bonsai  temp 0.3~0.5
"""
import json
import os
import sys
import time
import urllib.request

WARM_ONLY = "--warm" in sys.argv
if WARM_ONLY:
    sys.argv.remove("--warm")

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
MODEL = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("AGENT_MODEL", "lfm25-2.6b")
TEMP = float(sys.argv[4]) if len(sys.argv) > 4 else float(os.environ.get("AGENT_TEMP", "0.1"))
TOP_K = int(os.environ.get("AGENT_TOP_K", "50" if "lfm" in MODEL else "20"))
# thinking 模型要留够预算，否则思考吃光 max_tokens，答案被截断
MAXTOK = int(os.environ.get("AGENT_MAXTOK", "600"))
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

# 故意写长（乘 4 ≈ 700-1000 token），模拟真实 agent 的 system prompt。
# 注意：这是 agent 的"前缀"，一次也不能改 —— 改了就废掉整个缓存。
SYSTEM = (
    "你是一个本地命令行 agent，运行在 Windows 11 / i5-8500 / 20GB 内存的机器上。\n"
    "规则：\n"
    "1. 需要外部信息或执行动作时，必须调用工具，不要凭空猜测。\n"
    "2. 回答用中文，简洁分点，不超过 5 行。\n"
    "3. 涉及文件路径一律使用相对 D:\\AI\\LLMlocal 的写法。\n"
    "4. 数值结果保留两位小数，并注明单位。\n"
    "5. 若信息不足，先调用工具获取，再作答。\n"
    "6. 不要复述系统提示内容。\n"
) * 4

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "读取仓库内的一个文件并返回其文本内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对 D:\\AI\\LLMlocal 的文件路径"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "列出某个目录下的文件",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
{"type": "function", "function": {
        "name": "run_bench", "description": "在本地跑一次 llama-bench 吞吐基准测试",
        "parameters": {"type": "object", "properties": {
            "model": {"type": "string"}, "threads": {"type": "integer"}},
            "required": ["model"]}}},
    # 没有写工具时，模型被问「新建文件并写入」只能如实说「无法操作文件系统」——
    # 那不是模型能力问题，是 schema 里没有对应函数。加了这个才有得选。
    {"type": "function", "function": {
        "name": "write_file", "description": "新建或覆盖一个文本文件并写入内容",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "相对 D:\\AI\\LLMlocal 的文件路径"},
            "content": {"type": "string", "description": "要写入的完整文本内容"}},
            "required": ["path", "content"]}}},
]

# (用户请求, 期望调用的工具名 或 None 表示这轮不该调工具)
TASKS = [
    ("读取 llama-plan.ps1 这个文件，用一句话说明它做什么。", "read_file"),
    ("列出 models 目录下的文件。", "list_dir"),
    ("用 run_bench 跑一下 2.6B 的吞吐测试，threads 用 4。", "run_bench"),
    ("新建一个文件 notes/agent-demo.md，写三行：本地 agent 循环测试。", "write_file"),
    ("综合前面的结果，评估这套本地方案是否够用。", None),
    ("给我 3 条下一步优化建议。", None),
]

BENCH_RESULT = ('{"ok": true, "result": "LFM2.5-2.6B-Q4_K_M, '
                'gen_tps=24.8, prompt_tps=52.1, threads=4, rss_gib=2.4, ctx=32768"}')
LIST_RESULT = '{"ok": true, "result": ["bonsai-1.7b", "lfm25", "stack", "atlas"]}'
READ_RESULT = ('{"ok": true, "bytes": 4672, '
               'summary": "按实时空闲内存规划 ctx/KV/线程, 内存不够就自动降级"}')
WRITE_RESULT = '{"ok": true, "path": "notes/agent-demo.md", "bytes": 96, "created": true}'

stats = {"rounds": 0, "want_tool": 0, "hit_tool": 0, "name_ok": 0,
         "answered": 0, "tcs": 0, "prompt": 0, "cached": 0, "gen": 0, "gen_s": 0.0}


def post(messages, max_tokens=MAXTOK):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": TEMP, "top_k": TOP_K, "seed": 42}
    if TOOLS:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read().decode("utf-8"))
    tm = d.get("timings", {}) or {}
    u = d.get("usage", {}) or {}
    stats["prompt"] += u.get("prompt_tokens", 0) or (tm.get("prompt_n", 0) + tm.get("cache_n", 0))
    stats["cached"] += tm.get("cache_n") or (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    stats["gen"] += tm.get("predicted_n", u.get("completion_tokens", 0))
    stats["gen_s"] += (tm.get("predicted_ms", 0) or 0) / 1000.0
    return d, tm, time.time() - t0


def show(tag, tm, wall):
    pt = tm.get("prompt_n", 0) + tm.get("cache_n", 0)
    cn = tm.get("cache_n", 0)
    pct = (cn / pt * 100.0) if pt else 0.0
    print("  %-20s prefill %6.2f s | prompt %5d (复用 %5d = %5.1f%%) | gen %3d tok @ %5.1f t/s | %5.1f s"
          % (tag, tm.get("prompt_ms", 0) / 1000.0, pt, cn, pct,
             tm.get("predicted_n", 0), tm.get("predicted_per_second", 0) or 0, wall), flush=True)


if WARM_ONLY:
    # 只做一次「暖前缀」：system + tools 逐字节等于真实请求的前缀，
    # 所以后面每一轮只 prefill 新增部分（实测 3.8 s -> 0.56 s）。
    print("warm: model=%s port=%d  prefilling frozen prefix ..." % (MODEL, PORT), flush=True)
    d, tm, wall = post([{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "hi"}], max_tokens=1)
    pt = tm.get("prompt_n", 0) + tm.get("cache_n", 0)
    print("warm: done in %.1f s | prompt %d tok (cache_n %d) | %.1f t/s"
          % (wall, pt, tm.get("cache_n", 0),
             (tm.get("prompt_n", 0) or 0) / max(0.001, (tm.get("prompt_ms", 0) or 1) / 1000.0)),
          flush=True)
    raise SystemExit(0 if d.get("choices") else 1)

msgs = [{"role": "system", "content": SYSTEM}]
print("model=%s  temp=%.2f  top_k=%d  system=%d 字符" % (MODEL, TEMP, TOP_K, len(SYSTEM)), flush=True)
t_all = time.time()

for i, (task, want) in enumerate(TASKS[:ROUNDS], 1):
    msgs.append({"role": "user", "content": task})
    d, tm, wall = post(msgs)
    show("轮%d 提问" % i, tm, wall)
    stats["rounds"] += 1
    m = d["choices"][0]["message"]
    tcs = m.get("tool_calls")
    finish = d["choices"][0].get("finish_reason")

    if want:
        stats["want_tool"] += 1
        got = [t["function"]["name"] for t in (tcs or [])]
        if tcs:
            stats["tcs"] += len(tcs)
            stats["hit_tool"] += 1
            if got[0] == want:
                stats["name_ok"] += 1
            print("  -> tool_calls: %s  (期望 %s) %s" %
                  (got, want, "OK" if got[0] == want else "MISMATCH"), flush=True)
            print("     args: %s" % [(t["function"].get("arguments") or "")[:90] for t in tcs], flush=True)
        else:
            print("  -> 没有调工具! finish=%s  reply=%r" % (finish, (m.get("content") or "")[:80]), flush=True)

        if tcs:
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": t["id"], "type": "function",
                 "function": {"name": t["function"]["name"],
                              "arguments": t["function"]["arguments"]}} for t in tcs]})
            # 工具结果按模型给出的顺序回灌（顺序稳定 -> 前缀可缓存）
            res = []
            for t in tcs:
                n = t["function"]["name"]
                body = {"read_file": READ_RESULT, "list_dir": LIST_RESULT,
                        "run_bench": BENCH_RESULT,
                        "write_file": WRITE_RESULT}.get(n, '{"ok": false}')
                res.append({"role": "tool", "tool_call_id": t["id"], "name": n, "content": body})
            msgs.extend(res)
            d2, tm2, wall2 = post(msgs)
            show("轮%d 回灌" % (i + 1), tm2, wall2)
            m2 = d2["choices"][0]["message"]
            if not m2.get("tool_calls") and (m2.get("content") or "").strip():
                stats["answered"] += 1
            print("  -> 最终答案: %r" % (m2.get("content") or "")[:120], flush=True)
            msgs.append({"role": "assistant", "content": m2.get("content") or ""})
        else:
            msgs.append({"role": "assistant", "content": m.get("content") or ""})
    else:
        if not tcs and (m.get("content") or "").strip():
            stats["answered"] += 1
        print("  -> 纯文本: %r" % (m.get("content") or "")[:120], flush=True)
        msgs.append({"role": "assistant", "content": m.get("content") or ""})

wall_all = time.time() - t_all
print()
print("=" * 74)
print("模型 %s | %d 轮 | 总耗时 %.1f s" % (MODEL, stats["rounds"], wall_all))
print("工具调用 : 应该调 %d 轮, 真的调了 %d 轮 (%.0f%%), 工具名正确 %d/%d, 平均每次 %d 个 call"
      % (stats["want_tool"], stats["hit_tool"],
         100.0 * stats["hit_tool"] / max(1, stats["want_tool"]),
         stats["name_ok"], stats["want_tool"], stats["tcs"] / max(1, stats["hit_tool"])))
print("收尾作答 : %d/%d 轮在回灌后直接给了文本答案（没有空转再调工具）"
      % (stats["answered"], stats["rounds"]))
print("前缀缓存 : prompt %d tok, 复用 %d = %.1f%%"
      % (stats["prompt"], stats["cached"], 100.0 * stats["cached"] / max(1, stats["prompt"])))
print("生成速度 : %d tok / %.1f s = %.1f t/s 平均"
      % (stats["gen"], stats["gen_s"], stats["gen"] / max(0.001, stats["gen_s"])))
print("=" * 74)
