#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fs_probe.py — 复现 pi 里 LFM 模型「写不了文件」的那个场景，并且真的落盘

跑法:
    PYTHONIOENCODING=utf-8 python fs_probe.py [端口] [模型alias] [temp] [--clean]

    python fs_probe.py 8080 lfm25-2.6b 0.1 --clean     # 2.6B（quality）
    python fs_probe.py 8083 lfm25-1.2b 0.1 --clean     # 1.2B（fast）

为什么需要这个脚本（2026-10-12 实测）:
    pi + lfm25-1.2b，用户说「新建一个1.txt 写入今天的天气」。
    模型 4 次调工具，path 全部写成目录（D:/user/Desktop），
    工具报 EEXIST / EISDIR，之后模型开始编「无法直接操作文件系统」的借口。

    **tool_test.py 3/3 通过完全掩盖了这个问题** —— 它只验证
    「选对工具名」，从不检查参数里有没有文件名，也不真的执行。
    冒烟测试通过 != 能干活。

设计要点:
  1. 工具 schema 照抄 pi 的形状（小写 read/write/edit/bash + path 参数）。
     模型对工具名的记忆来自微调数据，用 Claude 风格的 Read/Write 测不出来。
  2. 分「弱提示」和「明确提示」两组：
       弱提示 = pi 里的真实原话（只有文件名 / 「写入当前根目录」），
                不指望它选对工具，只记录它会跑去干什么。
       明确提示 = 给了绝对工作目录和文件名，**必须** write 且落盘。
  3. 真的执行 write/read/edit（沙箱内），落盘才算数。
  4. 系统提示里给出工作目录 —— pi 也会把 cwd 告诉模型，不给就是故意使坏。

沙箱: probe-sandbox/，所有相对路径都解析到这里，绝不碰真实文件。
退出码: 0 = 明确轮落盘成功且没有投降式回答；1 = 否则。
"""
import json
import os
import shutil
import sys
import time
import urllib.request

ARGS = [a for a in sys.argv[1:] if a != "--clean"]
CLEAN = "--clean" in sys.argv
PORT = int(ARGS[0]) if len(ARGS) > 0 else 8080
MODEL = ARGS[1] if len(ARGS) > 1 else "lfm25-2.6b"
TEMP = float(ARGS[2]) if len(ARGS) > 2 else 0.1
TOP_K = int(os.environ.get("AGENT_TOP_K", "50"))
# 2.6B 是 thinking 模型，思考 token 和答案共用 max_tokens。
# 给 600 时实测出现 finish_reason=length + content=''（整个回答为空），
# 看起来就像「不会调工具」。1400 是稳的值。
MAXTOK = int(os.environ.get("AGENT_MAXTOK", "1400"))
URL = "http://127.0.0.1:%d/v1/chat/completions" % PORT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "probe-sandbox"))

SYSTEM = (
    "你是一个运行在 Windows 上的命令行 agent。\n"
    "需要创建或修改文件时调用 write 工具，不要用 bash 绕。\n"
    "path 参数是完整文件路径，必须包含文件名。\n"
    "你的工作目录是 %s（Windows 绝对路径）。\n"
    "回答用中文，不超过 3 行。\n"
) % ROOT.replace("\\", "\\\\")

# 照抄 pi 的工具名与参数名
TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read the contents of a file at the given path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "The path to the file"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write", "description": "Write content to a file at the given path (overwrites if it exists).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "The path to the file"},
            "content": {"type": "string", "description": "The content to write"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit", "description": "Edit an existing file by replacing oldText with newText.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "edits": {"type": "array", "items": {"type": "object", "properties": {
                "oldText": {"type": "string"}, "newText": {"type": "string"}}}}},
            "required": ["path", "edits"]}}},
    {"type": "function", "function": {
        "name": "bash", "description": "Execute a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]

# (用户请求, 期望工具 or None=不检查, 组)
# 弱提示 = pi 会话里的原话；明确提示 = 给了绝对目录 + 文件名
CASES = [
    ("新建一个1.txt 写入今天的天气", None, "弱"),
    ("写入当前根目录", None, "弱"),
    ("创建文件", None, "弱"),
    ("就在 %s 里建个 notes.txt，写三行：本地 agent 循环测试。" % ROOT, "write", "明确"),
]

GIVEUP = ("无法", "不能", "做不到", "没有权限")


def post(msgs):
    body = {"model": MODEL, "messages": msgs, "max_tokens": MAXTOK,
            "temperature": TEMP, "top_k": TOP_K, "seed": 42,
            "tools": TOOLS, "tool_choice": "auto"}
    req = urllib.request.Request(URL, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read().decode("utf-8"))
    return d, time.time() - t0


def resolve(p):
    """把模型给的 path 解析到沙箱里，返回 (绝对路径, 错误或 None)"""
    if not isinstance(p, str) or not p.strip():
        return None, "EMPTY: path 是空的"
    p = p.strip().strip('"').strip("'")
    abspath = p if os.path.isabs(p) else os.path.join(ROOT, p)
    abspath = os.path.abspath(abspath.replace("/", os.sep))
    try:
        if os.path.commonpath([abspath, ROOT]) != ROOT:
            return None, "ESCAPE: 路径跳出沙箱"
    except ValueError:
        return None, "ESCAPE: 路径非法"
    return abspath, None


def run_write(args):
    p, err = resolve(args.get("path"))
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False), False
    if os.path.isdir(p):
        return json.dumps({"ok": False, "error": "EEXIST: %s 是目录，不是文件" % p}, ensure_ascii=False), False
    content = args.get("content")
    if not isinstance(content, str):
        return json.dumps({"ok": False, "error": "content 不是字符串"}, ensure_ascii=False), False
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    return json.dumps({"ok": True, "path": os.path.relpath(p, ROOT),
                       "bytes": len(content.encode("utf-8"))}, ensure_ascii=False), True


def run_read(args):
    p, err = resolve(args.get("path"))
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False)
    if os.path.isdir(p):
        return json.dumps({"ok": False, "error": "EISDIR: %s 是目录" % p}, ensure_ascii=False)
    if not os.path.isfile(p):
        return json.dumps({"ok": False, "error": "ENOENT: %s 不存在" % p}, ensure_ascii=False)
    with open(p, encoding="utf-8") as f:
        return json.dumps({"ok": True, "content": f.read()[:2000]}, ensure_ascii=False)


def run_edit(args):
    p, err = resolve(args.get("path"))
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False)
    if os.path.isdir(p):
        return json.dumps({"ok": False, "error": "EISDIR: %s 是目录" % p}, ensure_ascii=False)
    if not os.path.isfile(p):
        return json.dumps({"ok": False, "error": "ENOENT: %s 不存在" % p}, ensure_ascii=False)
    with open(p, encoding="utf-8") as f:
        s = f.read()
    edits = args.get("edits")
    if isinstance(edits, dict):
        edits = [edits]
    for e in edits or []:
        s = s.replace(e.get("oldText", ""), e.get("newText", ""), 1)
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)
    return json.dumps({"ok": True, "path": os.path.relpath(p, ROOT)}, ensure_ascii=False)


def run_bash(args):
    """返回合理的假结果。返回 ok:false 会让模型拿到无信息反馈然后原地打转，
    那样测出来的是模型的耐心，不是参数保真度。"""
    c = (args.get("command") or "").strip().lower()
    if "date" in c:
        return json.dumps({"ok": True, "stdout": "Tue Oct 14 2026 晴 25C"}, ensure_ascii=False)
    if "ls" in c or c.startswith("dir"):
        return json.dumps({"ok": True, "stdout": "1.txt  notes  README.md"}, ensure_ascii=False)
    return json.dumps({"ok": True, "stdout": ""}, ensure_ascii=False)


RUNNERS = {"write": run_write, "read": run_read, "edit": run_edit}
SINGLE = {"bash": run_bash}

if CLEAN and os.path.isdir(ROOT):
    shutil.rmtree(ROOT)
os.makedirs(ROOT, exist_ok=True)

msgs = [{"role": "system", "content": SYSTEM}]
st = {"rounds": 0, "called": 0, "gave_up": 0, "trunc": 0,
      "clear_called": 0, "clear_name_ok": 0, "clear_path_ok": 0, "clear_landed": 0,
      "tool_hist": {}}

print("model=%s  port=%d  temp=%.2f  max_tokens=%d" % (MODEL, PORT, TEMP, MAXTOK))
print("sandbox=%s\n" % ROOT)

for i, (q, want, kind) in enumerate(CASES, 1):
    msgs.append({"role": "user", "content": q})
    d, wall = post(msgs)
    ch = d["choices"][0]
    m = ch["message"]
    tcs = m.get("tool_calls")
    st["rounds"] += 1
    print("[轮%d %s] %s" % (i, kind, q[:70]))
    if ch.get("finish_reason") == "length":
        st["trunc"] += 1
        print("   !! finish_reason=length —— 思考吃光了 max_tokens=%d，"
              "换个模型或调大 AGENT_MAXTOK" % MAXTOK)

    if not tcs:
        txt = (m.get("content") or "").strip()
        print("   没有调工具  finish=%s  %.1fs" % (ch.get("finish_reason"), wall))
        print("   回答: %s" % (txt[:150].replace("\n", " / ") or "(空)"))
        if any(k in txt for k in GIVEUP):
            st["gave_up"] += 1
            print("   !! 投降式回答 —— 模型在用嘴代替干活")
        msgs.append({"role": "assistant", "content": txt})
        print()
        continue

    st["called"] += 1
    if kind == "明确":
        st["clear_called"] += 1
    msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
        {"id": t["id"], "type": "function",
         "function": {"name": t["function"]["name"],
                      "arguments": t["function"]["arguments"]}} for t in tcs]})
    results = []
    for t in tcs:
        name = t["function"]["name"]
        st["tool_hist"][name] = st["tool_hist"].get(name, 0) + 1
        try:
            a = json.loads(t["function"]["arguments"] or "{}")
        except Exception:
            a = {}
        print("   -> %s(%s)" % (name, json.dumps(a, ensure_ascii=False)[:140]))
        if kind == "明确" and name == want:
            st["clear_name_ok"] += 1
        path = a.get("path")
        if isinstance(path, str) and path.strip():
            tail = path.strip().replace("\\", "/").rstrip("/").split("/")[-1]
            if "." in tail:
                if kind == "明确":
                    st["clear_path_ok"] += 1
            else:
                print("      !! path 里没有文件名: %r" % path)
        if name in RUNNERS:
            # write 返回 (结果, 是否真落盘)，read/edit 只返回结果
            r = RUNNERS[name](a)
            out, ok = r if isinstance(r, tuple) else (r, False)
            if name == "write" and ok:
                print("      OK 落盘: %s" % out)
                if kind == "明确":
                    st["clear_landed"] += 1
            else:
                print("      %s" % out[:150])
        else:
            out = SINGLE.get(name, lambda x: '{"ok":false}')(a)
        results.append({"role": "tool", "tool_call_id": t["id"], "name": name, "content": out})
    msgs.extend(results)
    print()

files = []
for root, _, fs in os.walk(ROOT):
    for f in fs:
        fp = os.path.join(root, f)
        files.append((os.path.relpath(fp, ROOT), os.path.getsize(fp)))

print("=" * 72)
print("沙箱里的文件:")
for n, sz in files or []:
    print("   %-40s %d bytes" % (n, sz))
if not files:
    print("   (空) —— 一轮都没落盘")
print("-" * 72)
print("全部 %d 轮: 调工具 %d | 投降式回答 %d | 被 max_tokens 截断 %d | 工具分布 %s"
      % (st["rounds"], st["called"], st["gave_up"], st["trunc"], st["tool_hist"]))
print("明确轮: 调工具 %d | 选对 write %d | path 带文件名 %d | 真落盘 %d"
      % (st["clear_called"], st["clear_name_ok"], st["clear_path_ok"], st["clear_landed"]))
ok = (st["clear_landed"] > 0 and st["gave_up"] == 0 and st["trunc"] == 0)
print("结论: %s" % ("PASS —— 能被指使着真的写文件" if ok else
                    "FAIL —— 别让它碰文件，或每次都人工确认 path"))
print("=" * 72)
sys.exit(0 if ok else 1)