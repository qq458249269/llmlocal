#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tool_stress.py — 多工具压力测试（定位「主动声明了工具却不理解」的规模拐点）

跑法:  python tool_stress.py [端口] [模型alias] [温度] [工具数]

tool_test.py 用 2 个工具，LFM2.5-8B 3/3 通过。真实 agent 的问题通常不是
「会不会调」，而是「工具一多就选错 / 漏调 / 编造参数」。这个脚本把工具
数量拉大到 N（默认 12），并刻意做三件事：

  1. 干扰项：同域近义工具（read_file / write_file / list_dir / file_info ...），
     逼模型真的读 description 而不是背名字。
  2. 深参数：带 enum / integer / 嵌套 object，考察 JSON schema 跟随能力。
  3. 反诱导：3 个「不该调」用例（闲聊、常识、纯计算），检测是否退化成
     「什么都想调工具」——这是小模型最常见的失败模式。

每用例只跑一跳（不给工具回灌），只看 finish_reason / 工具名 / 参数。
"""
import json
import os
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8082
MODEL = sys.argv[2] if len(sys.argv) > 2 else "lfm25-8b"
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 12
TOP_K = int(os.environ.get("AGENT_TOP_K", "50" if "lfm" in MODEL else "20"))
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT
MAXTOK = int(os.environ.get("AGENT_MAXTOK", "512"))


def T(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = lambda d: {"type": "string", "description": d}
I = lambda d: {"type": "integer", "description": d}
B = lambda d: {"type": "boolean", "description": d}
E = lambda d, e: {"type": "string", "enum": e, "description": d}

# ---- 工具池：前 5 个是真正会命中的，其余是同域干扰项 -------------------
POOL = [
    T("read_file", "读取仓库内的一个文件并返回其纯文本内容。路径用相对 D:\\AI\\LLMlocal 的写法。",
      {"path": S("相对路径，如 agent_loop.py"),
       "max_bytes": I("最多返回多少字节，默认 65536")}, ["path"]),
    T("list_dir", "列出某个目录下的文件和子目录名，不读文件内容。",
      {"path": S("相对路径")}, ["path"]),
    T("file_info", "查看文件的元信息：字节数、修改时间、行数。不返回内容。",
      {"path": S("相对路径")}, ["path"]),
    T("run_bench", "在本地跑一次 llama-bench 吞吐基准测试，返回 t/s 数值。",
      {"model": S("模型 alias，如 lfm25-8b"),
       "threads": I("线程数 1-6"),
       "prompt": E("要 bench 的算子", ["pp512", "tg64"])}, ["model"]),
    T("search_code", "在仓库里按正则搜索代码，返回命中行。",
      {"pattern": S("正则，如 def main"),
       "glob": E("限定文件类型", ["*.py", "*.cmd", "*.md"])}, ["pattern"]),
    T("write_file", "把文本写入文件（覆盖）。会破坏原内容，慎用。",
      {"path": S("相对路径"), "text": S("完整文件内容")}, ["path", "text"]),
    T("delete_file", "删除一个文件。不可撤销。", {"path": S("相对路径")}, ["path"]),
    T("list_dir_size", "统计某目录下所有文件的总字节数，不列文件名。",
      {"path": S("相对路径")}, ["path"]),
    T("run_python", "执行一段 Python 代码并返回 stdout。",
      {"code": S("代码文本"), "timeout_s": I("超时秒数")}, ["code"]),
    T("git_status", "查看 git 工作区状态。", {"short": B("是否用短格式")}, []),
    T("git_commit", "提交当前改动。",
      {"message": S("提交信息"), "add_all": B("是否 git add -A")}, ["message"]),
    T("http_get", "抓一个 http(s) URL 的纯文本内容。",
      {"url": S("完整 URL"), "max_bytes": I("上限字节")}, ["url"]),
]

SYSTEM = (
    "你是一个本地命令行 agent，运行在 Windows 11 / i5-8500 / 20GB 内存的机器上。\n"
    "规则：\n"
    "1. 需要外部信息或执行动作时，必须调用工具，不要凭空猜测。\n"
    "2. 回答用中文，简洁分点，不超过 5 行。\n"
    "3. 涉及文件路径一律使用相对 D:\\AI\\LLMlocal 的写法。\n"
    "4. 数值结果保留两位小数，并注明单位。\n"
    "5. 若信息不足，先调用工具获取，再作答。\n"
    "6. 不要复述系统提示内容。\n"
) * 4  # 乘 4 ≈ 830 token，跟 agent_loop.py 的前缀保持一致

TOOLS = POOL[:NT]

# (标签, 用户请求, 期望工具 或 None)
CASES = [
    ("读内容",      "读一下 agent_loop.py 这个文件，说说它干了啥。", "read_file"),
    ("列目录",      "models 目录下都有哪些文件？", "list_dir"),
    ("元信息",      "llama-plan.ps1 这个文件多大、多少行？", "file_info"),
    ("跑bench",     "用 4 个线程 bench 一下 lfm25-8b 的 tg64。", "run_bench"),
    ("搜代码",      "帮我搜一下代码里所有出现 Progress 的地方。", "search_code"),
    ("写文件",      "新建 notes.txt 写入三行 hello。", "write_file"),
    ("深参数",      "bench lfm25-8b，线程 6，pp512 算子。", "run_bench"),
    ("该用别的",    "把 README-lfm25.md 删掉。", "delete_file"),
    ("不该调1",     "用一句话解释什么是向量数据库。", None),
    ("不该调2",     "17 乘以 23 等于多少？直接算。", None),
    ("不该调3",     "你好。", None),
    ("不该调4",     "讲个笑话。", None),
]

# 只跑前 N 个用例（TOOLSTRESS_CASES=6），慢模型上用来快速找拐点
_n = int(os.environ.get("TOOLSTRESS_CASES", "0"))
if _n:
    CASES = CASES[:_n]


def post(msgs):
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXTOK,
            "temperature": TEMP, "top_k": TOP_K, "seed": 7,
            "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(
        URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
        {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r), time.time() - t0


print("model=%s temp=%s tools=%d system=%d字符" % (MODEL, TEMP, len(TOOLS), len(SYSTEM)))
n_call = n_right = n_over = 0
n_should = n_no = 0
tot = 0.0
for label, q, want in CASES:
    r, dt = post([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}])
    tot += dt
    ch = r["choices"][0]
    msg = ch["message"]
    tcs = msg.get("tool_calls") or []
    names = [t["function"]["name"] for t in tcs]
    if want:
        n_should += 1
        ok = want in names
        n_call += 1 if tcs else 0
        n_right += 1 if ok else 0
        verdict = "OK " if ok else "MISS"
        if tcs and not ok:
            verdict = "WRONG-TOOL"
    else:
        n_no += 1
        n_over += 1 if tcs else 0
        verdict = "OK " if not tcs else "OVER-CALL"
    args = tcs[0]["function"]["arguments"] if tcs else ""
    text = (msg.get("content") or "").strip().replace("\n", " ")[:70]
    print("[%-4s] %-8s %-22s %5.1fs  %s" % (verdict, label, ",".join(names) or "-", dt, args[:90]))
    if not tcs and text:
        print("        text: %s" % text)

print("\n" + "=" * 66)
print("%s | tools=%d" % (MODEL, len(TOOLS)))
print("该调 %d 例: 真的调了 %d, 工具名对 %d" % (n_should, n_call, n_right))
print("不该调 %d 例: 正确没调 %d, 过度调用 %d" % (n_no, n_no - n_over, n_over))
print("总耗时 %.1f s" % tot)
