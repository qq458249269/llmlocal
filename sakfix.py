#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sakfix.py — SakThai 0.5B 的输出归一化反向代理

    python sakfix.py [上游端口=8084] [监听端口=8085]

问题（实测，sak_diag.py）：SakThai 0.5B 能选对工具、参数也对，但它输出的
包装标签每次都不一样，llama.cpp 的解析器只认其中一种：

    <tool_call>{"name":...}</tool_call>       唯一被原生解析
    <tools>{"type":"function","function":{...}}</tools>   0/8
    {"name":"list_dir","arguments":{...}}                0/8
    tools[name: 'list_dir', arguments: {...}]           0/8
    <tool>list_dir</tool>{"arguments":{...}}            0/8（这是模型卡的训练格式）
    FUNCTION: list_dir\nPARAMETERS: {...}               0/8

所以 tool_calls 恒为 null，看起来像「模型不支持工具调用」。

这个代理做三件事，全在客户端，不改模型不改 server：
  1. 截断废话：按对话轮次标记（user/assistant/system/tool/<|im_end|>）切掉
     自问自答的后半段，solve 「循环说废话」
  2. 格式归一：把上面 5 种形态统一抠成 OpenAI 的 tool_calls
  3. 合法性校验：工具名不在请求给的 tools 里就丢弃（防幻觉工具名），
     必填参数缺失也丢弃（防「arguments":{}」这种 schema 回显）

开关：--no-oneshot 关掉格式预热，--no-repair 关掉「解析不出来时的 JSON-Schema 修复回合」。
用法：pi / 测试脚本指向 8085，上游 llama-server 仍然跑在 8084。
不支持 SSE 流式（强制 stream=false）。

默认还会自动注入 one-shot 格式示例（--no-oneshot 关掉）：0.5B 记不住规则，
但照抄示例很行；示例内容只由请求里的 tools 决定，逐字节固定 → 仍能命中前缀缓存。
示例里特意带了「工具结果 → 中文一句话收尾」这一轮，把「答完就停」也教了。
"""
import ast
import json
import os
import re
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = int(sys.argv[1]) if len(sys.argv) > 1 else 8084
LISTEN = int(sys.argv[2]) if len(sys.argv) > 2 else 8085
ONESHOT = "--no-oneshot" not in sys.argv
REPAIR = "--no-repair" not in sys.argv
UP_URL = "http://127.0.0.1:%d" % UP

# 轮次标记：模型自问自答的起点，也是废话的起点
TURN_RE = re.compile(r"(<\|im_end\|>|<\|endoftext\|>|\nuser\n|\nassistant\n|"
                     r"\nsystem\n|\nuser:|\nassistant:|\n工具结果)", re.I)
# 5 种已知的野格式
RE_TOOLS_BLOCK = re.compile(r"<tools>(.*?)</tools>", re.S | re.I)
RE_TOOL_NAME_TAG = re.compile(r"<tool>\s*([A-Za-z0-9_.]+)\s*</tool>", re.I)
RE_FUNC_KV = re.compile(r"FUNCTION\s*:\s*([A-Za-z0-9_.]+)\s*\n+\s*PARAMETERS\s*:\s*", re.I)
RE_PY_TOOLS = re.compile(r"tools\s*\[(.*?)\]", re.S | re.I)
KW_RULES = [(r"weather|天气|温度|下雨|气温", r"weather"),
            (r"time|clock|date|几点|时间|日期", r"time"),
            (r"bash|shell|command|命令|执行|脚本", r"bash|shell|exec|run"),
            (r"write|save|create|edit|写|保存|新建", r"write|save|create|edit|put"),
            (r"read|open|get_file|读|读取|查看文件", r"read|open|file|get_"),
            (r"dir|list|ls|目录|列出", r"dir|list|ls")]
_CALL_HINT_PAT = [r"<functions?>?\s*call\s*</functions?>", r"函数调用", r"工具调用",
                   r"tool[_ ]?call", r"调用.{0,6}工具", r"使用.{0,6}工具",
                   r"\{\s*\"name\"\s*:", r"<tools>"]
RE_CALL_HINT = re.compile("|".join(_CALL_HINT_PAT), re.I)


def resolve_by_keyword(text, tools):
    """模型想调工具但没点名时，用关键词猜。命中必须唯一，否则不猜。"""
    names = list(tools)
    for pat, want in KW_RULES:
        if not re.search(pat, text, re.I):
            continue
        hit = [n for n in names if re.search(want, n, re.I)]
        if len(hit) == 1:
            return hit[0]
    return names[0] if len(names) == 1 else None


RE_CN_CALL = re.compile(r"(?:<functions?>\s*call\s*</functions?>\s*)?"
                        r"(?:以|用|调用|使用|通过)\s*([A-Za-z0-9_.]+)\s*"
                        r"(?:函数|工具|功能|function|tool)", re.I)


def _schema_args(spec, sample):
    """按 schema 的 required 造一份合法参数（占位值）"""
    props = (spec or {}).get("parameters", {}).get("properties", {}) or {}
    need = (spec or {}).get("parameters", {}).get("required", []) or []
    out = {}
    for k in (need or list(props)[:1]):
        t = (props.get(k) or {}).get("type", "string")
        out[k] = {"string": sample, "integer": 5, "number": 5.0,
                  "boolean": True}.get(t, sample)
    return out


# one-shot 示例：按工具名猜一个语义匹配的例句，固定不变 → 前缀缓存照样命中
_DEMO = [
    (r"dir|list|ls", "看看 models 目录里有哪些文件", {"path": "models"}, "list_dir",
     "models 目录里有 12 个 GGUF 文件。", "有 12 个 GGUF 文件。"),
    (r"read|open|cat|file", "读一下 README.md 的开头", {"path": "README.md"}, "read_file",
     "# LLMlocal\n纯 CPU 本地大模型推理工作台。", "README.md 讲的是本地推理工作台。"),
    (r"write|save|create|edit", "在 notes/demo.txt 写入一行测试",
     {"path": "notes/demo.txt", "content": "测试"}, "write_file",
     "OK 落盘: notes/demo.txt (6 bytes)", "已经把「测试」写进 notes/demo.txt 了。"),
    (r"weather", "上海现在天气怎么样？用摄氏度", {"city": "上海", "unit": "c"}, "get_weather",
     '{"temp_c": 27.5, "condition": "多云"}', "上海 27.5 度，多云。"),
    (r"time|clock|date", "现在几点？", {}, "get_time",
     '{"tz": "Asia/Shanghai", "time": "14:20"}', "现在是 14:20。"),
]


def build_oneshot(tools):
    """给 0.5B 看的格式样板：调用 -> 工具结果 -> 一句话收尾（教它停）"""
    if not tools:
        return []
    first = list(tools.values())[0]
    name = first.get("name", "")
    for pat, q, args, tname, tres, tans in _DEMO:
        if re.search(pat, name, re.I) and tname in tools:
            a = args
            break
    else:
        q, tname = "请帮我处理一下这个请求", name
        a = _schema_args(first, "示例")
        tres, tans = "OK", "已经处理完了。"
    if tname not in tools:                      # 样板的工具不在本次清单里
        tname, a = name, _schema_args(first, "示例")
    return [
        {"role": "user", "content": q},
        {"role": "assistant", "tool_calls": [{"id": "call_demo", "type": "function",
                                              "function": {"name": tname,
                                                           "arguments": json.dumps(a,
                                                             ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_demo", "name": tname, "content": tres},
        {"role": "assistant", "content": tans},
    ]


def loads_lenient(s):
    """JSON 优先，失败就当 Python 字面量（模型爱用单引号）"""
    s = s.strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            v = fn(s)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
    return None


def first_json_obj(s):
    """抠出第一个配平的 {...}（模型经常在 JSON 后面接废话）"""
    start = s.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj = loads_lenient(s[start:i + 1])
                    if obj is not None:
                        return obj
                    break
        start = s.find("{", start + 1)
    return None


def extract_calls(text, tools):
    """从正文里抠出 [(name, arguments_dict)]，只保留通过校验的"""
    out, seen = [], set()

    def push(name, args):
        name = (name or "").strip().strip("\"'")
        if not name or name in seen:
            return
        # 幻觉工具名：不在请求给的清单里就丢
        if tools and name not in tools:
            return
        if args is None:
            args = {}
        if isinstance(args, str):
            args = loads_lenient(args) or {}
        if not isinstance(args, dict):
            return
        # 参数一个都没有 = 多半是 schema 回显或模型在念叨，不是真调用
        if not args:
            return
        args = clean_args(args)
        seen.add(name)
        out.append((name, args))

    m = RE_TOOLS_BLOCK.search(text)
    if m:                                          # <tools>{...}</tools>
        inner = m.group(1).strip()
        obj = first_json_obj(inner)
        if obj:
            fn = obj.get("function", obj)          # OpenAI 的 {"type":"function","function":{…}}
            push(fn.get("name"), fn.get("arguments"))
    m = RE_PY_TOOLS.search(text)                    # tools[name: 'x', arguments: {...}]
    if m:
        inner = m.group(1).strip()
        nm = re.search(r"""name\s*:\s*['"]([^'"]+)['"]""", inner)
        am = re.search(r"arguments\s*:\s*(\{.*)", inner, re.S)
        push(nm.group(1) if nm else None,
             loads_lenient(am.group(1).split("}")[0] + "}") if am else None)
    m = RE_FUNC_KV.search(text)                      # FUNCTION: x\nPARAMETERS: {...}
    if m:
        rest = text[m.end():]
        obj = first_json_obj(rest) or {}
        push(m.group(1), obj)
    m = RE_TOOL_NAME_TAG.search(text)                # <tool>x</tool>{"arguments":{…}}
    if m:
        rest = text[m.end():]
        obj = first_json_obj(rest) or {}
        push(m.group(1), obj.get("arguments", obj if "arguments" not in obj else None))
    m = RE_CN_CALL.search(text)                      # <functions>call</functions> 以get_weather函数调用
    if m:
        rest = text[m.end():]
        obj = first_json_obj(rest) or {}
        spec = (tools or {}).get(m.group(1), {})
        props = set((spec.get("parameters") or {}).get("properties") or {})
        args = obj if props and (set(obj) & props) else None
        push(m.group(1), args)
    # 有调用意图但没点名工具：不在这里硬造调用。空参数既可能是 schema 回显，
    # 也可能是真调用，硬猜出来的路径比不调用更危险 —— 交给 repair 回合，
    # 参数让模型自己填（关键词只作为工具名提示传进去）。
    obj = first_json_obj(text)                       # 裸 {"name":…,"arguments":…}
    if obj:
        fn = obj.get("function", obj)
        if fn.get("name"):
            push(fn.get("name"), fn.get("arguments"))
    return out


def sanitize_history(msgs, tools):
    """把历史里的 tool_calls.arguments 修成合法 JSON（llama.cpp 渲染模板前会硬解析）。"""
    tools = tools or {}
    fixed = dropped = 0
    for m in msgs:
        if not isinstance(m, dict):
            continue
        for t in m.get("tool_calls") or []:
            fn = (t or {}).get("function") or {}
            name, raw = fn.get("name"), fn.get("arguments")
            if not name or name not in tools:
                continue
            if isinstance(raw, dict):
                raw = json.dumps(raw, ensure_ascii=False)
                fn["arguments"] = raw
            if not isinstance(raw, str):
                fn["arguments"] = "{}"
                dropped += 1
                continue
            if loads_lenient(raw) is not None:
                continue                       # 本来就合法（单引号也行）
            spec = tools[name].get("parameters") or {}
            props = list((spec.get("properties") or {}).keys())
            ok = None
            # 死法一：字符串被 max_tokens 截断，未闭合引号（也常常连右花括号都没有）
            for quote in ('"', "'"):
                for tail in (quote, quote + "}", quote + "}}"):
                    try:
                        cand = json.loads(raw.rstrip() + tail)
                        if isinstance(cand, dict) and cand:
                            ok = cand
                            break
                    except Exception:
                        pass
                if ok:
                    break
            # 死法二：多写了几笔，抠出第一个配平的 {...} 就收手
            if ok is None:
                obj = first_json_obj(raw)
                if obj:
                    ok = obj
            if ok is None:                     # 实在救不回来：只保留能认出的字段名
                ok = {k: None for k in props[:2]} or {"_truncated": True}
                dropped += 1
            else:
                fixed += 1
            if not m.get("content"):
                # 让模型仍看得到自己写歪的那段（截断版），而不是凭空消失
                m["content"] = "[上一轮调用参数被截断，原文]: " + raw[:200]
            fn["arguments"] = json.dumps(ok, ensure_ascii=False)
    if fixed or dropped:
        sys.stderr.write("[sakfix] 历史 tool_calls 修复: 补全 %d, 降级 %d" % (fixed, dropped))
        sys.stderr.flush()
    return fixed + dropped


def repair_call(req_body, tools, hint=None):
    """修复回合：把「我需要调用工具但没给全格式」逼成合法 JSON。
    用 server 的 json_schema（转 GBNF）约束，name 限定在本次 tools 里。"""
    NL = chr(10)
    lines = []
    for n, spec in tools.items():
        props = (spec.get("parameters") or {})
        lines.append("  %s(%s) —— %s" % (
            n, ", ".join((props.get("properties") or {}).keys()) or "无参数",
            (spec.get("description") or "")[:40]))
    last_user = ""
    for m in reversed(req_body.get("messages") or []):
        if m.get("role") == "user":
            last_user = m.get("content") or ""
            break
    body = {
        "model": req_body.get("model", "sakthai-0.5b"),
        "messages": [
            {"role": "system",
             "content": "你是一个只输出 JSON 的接口，不解释、不加标签。"},
            {"role": "user", "content":
             ("模型大概想调用：" + hint + NL + NL if hint else "") +
             "可用工具:" + NL + NL.join(lines) + NL + NL +
             "用户请求：" + NL + last_user + NL + NL +
             "输出这个工具调用的 JSON（name 必须是上面清单里的，arguments 按 schema 填）。"},
        ],
        "max_tokens": 96, "temperature": 0.1, "top_k": 20, "stream": False,
        "json_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(tools)},
                           "arguments": {"type": "object"}},
            "required": ["name", "arguments"]},
    }
    req = urllib.request.Request(UP_URL + "/v1/chat/completions",
                                 json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
    except Exception:
        return None
    obj = first_json_obj((d.get("choices") or [{}])[0].get("message", {}).get("content") or "")
    if not obj:
        return None
    fn = obj.get("function", obj)
    name, args = fn.get("name"), fn.get("arguments")
    if not name or name not in tools:
        return None
    if isinstance(args, str):
        args = loads_lenient(args) or {}
    return name, args if isinstance(args, dict) else {}


REPEAT_CHAR = re.compile(r"(.{12,}?)\1{6,}", re.S)    # 同一段 12 字以上连写 7 次
REPEAT_UNIT = re.compile(r"([\u4e00-\u9fffA-Za-z0-9]{1,12}?)\1{5,}|(.)\2{39,}", re.S)


def cut_repetition(text):
    """砍掉退化重复，只留一个单元（复读 5 次以上 / 同字符连写 40 个以上）。"""
    if not text or len(text) < 24:
        return text
    for rx in (REPEAT_CHAR, REPEAT_UNIT):
        m = rx.search(text)
        if not m:
            continue
        unit = m.group(1) or m.group(2) or ""
        # 从句中开始：留前缀 + 一个单元；从句首开始：只留一个单元（否则整句被砍没）
        text = text[:m.start()] + unit if m.start() > 0 else unit
    return text


def clean_args(obj):
    """参数里的长字符串同样切重复，否则会把 max_tokens 吃光 -> arguments 截断 -> 下轮 500。"""
    if isinstance(obj, str):
        return cut_repetition(obj)
    if isinstance(obj, dict):
        return {k: clean_args(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_args(v) for v in obj]
    return obj


def cut_ramble(text):
    text = cut_repetition(text)
    m = TURN_RE.search(text)
    if m and m.start() > 0:
        return text[:m.start()].rstrip()
    return text.rstrip()



class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _err(self, msg):
        sys.stderr.write("[sakfix] %s" % (msg,))
        sys.stderr.flush()

    def log_message(self, fmt, *a):
        self._err("%s - %s" % (self.address_string(), fmt % a))

    def _relay(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if self.path.startswith("/v1/chat/completions"):
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            # 强制非流式 + 客户端侧先加 stop，省得废话白生成几百 token
            body["stream"] = False
            msgs = body.get("messages") or []
            tools = {t["function"]["name"]: t["function"]
                     for t in (body.get("tools") or []) if t.get("type") == "function"}
            if msgs and not tools:
                tools = None                                   # 纯问答，不做调用解析
            elif ONESHOT and tools:
                # 格式预热：插在最前面，内容只由 tools 决定 → 前缀缓存仍命中
                demo = build_oneshot(tools)
                if msgs and msgs[0].get("role") == "system":
                    msgs = [msgs[0]] + demo + msgs[1:]
                else:
                    msgs = demo + msgs
                body["messages"] = msgs
            sanitize_history(msgs, tools)
            raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(UP_URL + self.path, raw,
                                         {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    d = json.load(r)
            except urllib.error.HTTPError as e:           # llama.cpp 自己报错：原文透传
                msg = e.read().decode("utf-8", "replace")
                open("sakfix_last_req.json", "wb").write(raw)
                open("sakfix_last_err.txt", "w", encoding="utf-8").write(msg)
                out = json.dumps({"error": {"message": msg, "type": "upstream",
                                    "code": e.code}}, ensure_ascii=False).encode("utf-8")
                self.send_response(e.code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                self._err("upstream %d: %s" % (e.code, msg[:400]))
                return
            except Exception as e:                            # 连不上才回 502
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                sys.stderr.write("[sakfix] upstream error: %r" % (e,))
            self._fix(d, tools, body)
            out = json.dumps(d, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        req = urllib.request.Request(UP_URL + self.path, raw, dict(self.headers))
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                out = r.read()
                self.send_response(r.status)
                self.send_header("Content-Type", r.headers.get("Content-Type",
                                                               "application/json"))
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
        except urllib.error.HTTPError as e:
            out = e.read()
            self.send_response(e.code)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

    def _fix(self, d, tools, body=None):
        if not tools:
            return
        for ch in d.get("choices", []):
            m = ch.get("message") or {}
            if m.get("tool_calls"):                          # 上游已解析出来，原样放行
                continue
            text = cut_ramble(m.get("content") or "")
            calls = extract_calls(text, tools)
            if not calls and REPAIR and RE_CALL_HINT.search(text) and body:
                rc = repair_call(body, tools, resolve_by_keyword(text, tools))
                if rc:
                    calls = [rc]
            if not calls:
                m["content"] = text
                continue
            m["tool_calls"] = [
                {"id": "call_sakfix_%d" % i, "type": "function",
                 "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                for i, (n, a) in enumerate(calls)]
            m["content"] = ""
            ch["finish_reason"] = "tool_calls"

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


if __name__ == "__main__":
    print("[sakfix] http://127.0.0.1:%d/v1  ->  upstream %s" % (LISTEN, UP_URL))
    print("[sakfix] 把 pi / 测试脚本的 base_url 指到 %d，原生模型端口仍是 %d" % (LISTEN, UP))
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()