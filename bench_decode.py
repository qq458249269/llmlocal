#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bench_decode.py — decode 速度 vs 上下文长度的曲线

跑法:  python bench_decode.py [端口] [模型alias] [长度列表]

agent 的每一轮都要读整条 KV，所以 decode t/s 随「当前对话有多长」下降，
而不是随 -c 上限变化。这个脚本对每个目标长度打两次请求：
    max_tokens=1    -> 纯 prefill，测出 prefill 速度
    max_tokens=161  -> 同样的 prefill + 160 tok 生成
两次的差就是 decode 速度，不依赖 server 日志。
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
MODEL = sys.argv[2] if len(sys.argv) > 2 else "probe"
LENS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                          else "1024,4096,8192,16384,24576".split(","))]
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

FILLER = ("这是一段用于填充上下文的普通说明文字，描述本地推理环境、内存规划、"
          "缓存策略和工具调用流程。") * 4

SYSTEM = ("你是一个本地命令行 agent。规则：\n"
          "1. 需要外部信息或执行动作时，必须调用工具，不要凭空猜测。\n"
          "2. 回答用中文，简洁分点，不超过 5 行。\n"
          "3. 数值结果保留两位小数，并注明单位。\n"
          "4. 不要复述系统提示内容。\n") * 4

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "读取仓库内的一个文件并返回其纯文本内容。",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "相对路径"}}, "required": ["path"]}}}]


def post(msgs, max_tokens):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "temperature": 0.2, "top_k": 80, "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r), time.time() - t0


print("%-8s %-9s %-14s %-14s" % ("目标", "实际prompt", "prefill", "decode(160tok)"))
for target in LENS:
    # FILLER 每段实测 ~130 token（中文），用它把 KV 撑到目标长度
    n_filler = max(1, int(target / 130))
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": FILLER * n_filler},
            {"role": "assistant", "content": "好的，我已读取背景。"},
            {"role": "user", "content": "用一句话总结上面的背景，然后停。"}]
    # 第一次纯 prefill（冷，测 prefill 速度），第二次同 prompt + 生成（走 LCP 缓存）
    r1, t1 = post(msgs, 1)
    r2, t2 = post(msgs, 161)
    ntok = r2["usage"]["prompt_tokens"]
    gen = r2["usage"]["completion_tokens"]
    pf = ntok / max(t1, 1e-6)
    dec = (t2 - t1) / max(gen - r1["usage"]["completion_tokens"], 1)
    print("%-8d %-9d %-14s %.1f t/s" % (target, ntok, "%.1f t/s" % pf, dec))
