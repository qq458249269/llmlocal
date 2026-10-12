#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sak_unit.py -- sakfix.py 归一化逻辑的单测（不碰模型，秒级）

    python sak_unit.py
"""
import importlib.util
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

spec = importlib.util.spec_from_file_location("sakfix", "sakfix.py")
S = importlib.util.module_from_spec(spec)
sys.argv = ["sakfix.py"]            # 模块顶层会读 argv 取端口
spec.loader.exec_module(S)

TOOLS = {
    "list_dir": {"name": "list_dir", "description": "列出目录下的文件",
                 "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                "required": ["path"]}},
    "read_file": {"name": "read_file", "description": "读取某个文件",
                  "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                 "required": ["path"]}},
    "get_weather": {"name": "get_weather", "description": "查天气",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"},
                                                  "unit": {"type": "string"}}}},
    "get_time": {"name": "get_time", "description": "查时间", "parameters": {}},
    "write_file": {"name": "write_file", "description": "写文件",
                  "parameters": {"type": "object",
                                 "properties": {"path": {"type": "string"},
                                                "content": {"type": "string"}},
                                 "required": ["path", "content"]}},
}

# (名字, 模型原文, 期望的 calls)
CASES = [
    ("原生 tool_call 之外的内层形状",
     '<tools>{"type":"function","function":{"name":"list_dir","arguments":{"path":"models"}}}'
     '</tools>',
     [("list_dir", {"path": "models"})]),
    ("裸 JSON",
     '{"name":"read_file","arguments":{"path":"README.md"}}',
     [("read_file", {"path": "README.md"})]),
    ("Python 字面量",
     "tools[name: 'list_dir', arguments: {'path': 'models'}]",
     [("list_dir", {"path": "models"})]),
    ("模型卡训练格式 <tool>name</tool>",
     '<tool>list_dir</tool>{"arguments":{"path":"models"}}',
     [("list_dir", {"path": "models"})]),
    ("FUNCTION/PARAMETERS",
     "FUNCTION: list_dir\nPARAMETERS: {\"path\": \"models\"}",
     [("list_dir", {"path": "models"})]),
    ("单引号 + 尾随废话",
     "<tool_call>{'name': 'get_weather', 'arguments': {'city': '上海'}}</tool_call> 好的，已经帮你查了。",
     [("get_weather", {"city": "上海"})]),
    ("arguments 是字符串",
     '{"name":"get_weather","arguments":"{\\"city\\":\\"上海\\"}"}',
     [("get_weather", {"city": "上海"})]),
    ("JSON 后面接自问自答（先截断再解析）",
     '{"name":"list_dir","arguments":{"path":"models"}}\nuser: 再看看 README\nassistant: 好的',
     [("list_dir", {"path": "models"})]),
    ("幻觉工具名 -> 丢弃", '{"name":"rm_rf","arguments":{"path":"/"}}', []),
    ("空 arguments 的 schema 回显 -> 丢弃",
     '{"name":"get_time","arguments":{}}', []),
    ("纯中文回答 -> 不硬造调用",
     "models 目录里有 9 个 GGUF 文件。", []),
]

# 截断测试
CUTS = [
    ("好了，这是结果。\nuser: 继续\nassistant: 好的", "好了，这是结果。"),
    ("模型单轮自答<|im_end|>", "模型单轮自答"),
    ("直接答完，没有废话", "直接答完，没有废话"),
]


def main():
    bad = 0
    for label, text, want in CASES:
        got = S.extract_calls(S.cut_ramble(text), TOOLS)
        ok = got == want
        bad += not ok
        print("%s %-34s -> %s" % ("PASS" if ok else "FAIL", label, got))
    for text, want in CUTS:
        got = S.cut_ramble(text)
        ok = got == want
        bad += not ok
        print("%s cut_ramble %-34s -> %r" % ("PASS" if ok else "FAIL", text[:30], got))
    # one-shot：必须只依赖 tools，且逐字节固定（否则前缀缓存全废）
    a = json.dumps(S.build_oneshot(TOOLS), ensure_ascii=False, sort_keys=True)
    b = json.dumps(S.build_oneshot(TOOLS), ensure_ascii=False, sort_keys=True)
    ok = a == b and "call_demo" in a
    bad += not ok
    print("%s build_oneshot 稳定且含 tool_calls" % ("PASS" if ok else "FAIL"))
    demo = S.build_oneshot(TOOLS)
    ok = demo and demo[-1]["role"] == "assistant" and not demo[-1].get("tool_calls")
    bad += not ok
    print("%s build_oneshot 末轮是「一句话收尾」（教它停）" % ("PASS" if ok else "FAIL"))
# 有调用意图但没点名工具：不硬造调用，只把工具名当提示（交给 repair 回合）
    txt = "我需要调用工具来查询天气"
    got = S.extract_calls(txt, TOOLS)
    nm = S.resolve_by_keyword(txt, TOOLS)
    ok = got == [] and nm == "get_weather" and bool(S.RE_CALL_HINT.search(txt))
    bad += not ok
    print("%s 有意图不硬造调用 天气->提示 %-6s calls=%s" %
          ("PASS" if ok else "FAIL", nm, got))
    # 历史 tool_calls 的 arguments 被 max_tokens 截断（未闭合字符串）
    # -> llama.cpp 渲染模板时硬解析失败，直接 500（重试/降参数都救不了，只能客户端修）
    trunc = '{"path": "notes/a.md", "content": "第一行\\n- ' + "0" * 400
    assert S.loads_lenient(trunc) is None
    bad_msg = {"role": "assistant", "content": None,
           "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "write_file", "arguments": trunc}}]}
    good = {"role": "assistant", "content": "好的", "tool_calls": [
        {"id": "c2", "type": "function", "function": {
            "name": "read_file", "arguments": '{"path": "README.md"}'}}]}
    hist = [bad_msg, good]
    n = S.sanitize_history(hist, TOOLS)
    a = json.loads(hist[0]["tool_calls"][0]["function"]["arguments"])
    ok = n >= 1 and a.get("path") == "notes/a.md" and isinstance(a.get("content"), str)
    ok = ok and json.loads(hist[1]["tool_calls"][0]["function"]["arguments"]) == {"path": "README.md"}
    ok = ok and bool(hist[0].get("content"))
    bad += not ok
    bad += not ok
    print("%s 历史截断 arguments 修成合法 JSON  -> %s" % ("PASS" if ok else "FAIL", a))
    # 修不好的（连字段都认不出）也不能让模板炸
    junk = {"role": "assistant", "tool_calls": [
        {"id": "c3", "type": "function", "function": {"name": "list_dir", "arguments": "完全不是JSON"}}]}
    S.sanitize_history([junk], TOOLS)
    ok = S.loads_lenient(junk["tool_calls"][0]["function"]["arguments"]) is not None
    bad += not ok
    print("%s 彻底认不出的 arguments 降级成合法 JSON" % ("PASS" if ok else "FAIL"))
    # 幻觉工具名的历史调用：llama.cpp 不在 tools 里就不管，但不能改成假的
    keep = {"role": "assistant", "tool_calls": [
        {"id": "c4", "type": "function", "function": {"name": "rm_rf", "arguments": "{bad"}}]}
    S.sanitize_history([keep], TOOLS)
    ok = keep["tool_calls"][0]["function"]["arguments"] == "{bad"
    bad += not ok
    print("%s 不在 tools 里的历史调用保持原样    " % ("PASS" if ok else "FAIL"))

    # 退化重复：0.5B 写长 content 时会 "000000…" 几百个，吃光 max_tokens
    rep = S.cut_repetition("第一行" + "0" * 400)
    ok = len(rep) <= 8
    bad += not ok
    print("%s 退化重复(同字符) 截短              -> %r" % ("PASS" if ok else "FAIL", rep))
    rep2 = S.cut_repetition("总结一下" + "总结一下" * 20)
    ok = len(rep2) < 40
    bad += not ok
    print("%s 退化重复(短语复读) 截短            -> %r" % ("PASS" if ok else "FAIL", rep2))
    ok = S.cut_repetition("短句子里没有重复") == "短句子里没有重复"
    bad += not ok
    print("%s 正常句子不动" % ("PASS" if ok else "FAIL"))
    ca = S.clean_args({"path": "a.md", "content": "x" + "0" * 300})
    ok = len(ca["content"]) < 20 and ca["path"] == "a.md"
    bad += not ok
    print("%s clean_args 递归切重复              -> %r" % ("PASS" if ok else "FAIL", ca["content"][:20]))

    total = len(CASES) + len(CUTS) + 12
    print("\n%s  (%d/%d passed)" % ("ALL PASS" if not bad else "%d FAILED" % bad,
                                    total - bad, total))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())