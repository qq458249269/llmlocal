#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_client.py — 面向 llama.cpp 前缀缓存的 OpenAI 兼容客户端封装

为什么需要它：CPU 上 prefill 贵，而 llama.cpp 的 prompt cache 是
「按 token 序列找最长公共前缀」。所以调用姿势比模型参数更影响体感：

  实测（i5-8500 4 核，Ternary-Bonsai-1.7B-PQ2_0，128k/q4_0）
    2275 token 的 system prompt 冷启动 ....... TTFT 82.6 s
    同样 prompt 第二次 ..................... TTFT  0.67 s (99.2% 复用)
    6 轮真实 agent 循环( system + 3 个工具 ) . 每轮 TTFT 1.5-2.5 s，复用 90-94%

四条铁律（本模块帮你兜住 1~3）：
  1) system prompt 与 tools 数组放最前，且逐字节不变
     —— 千万别在里面插「当前时间 / 会话 ID / 随机数」，一次就废掉整个前缀
  2) tools 的顺序与数量不能变（工具 schema 是 prompt 的一部分）
  3) 历史只追加、不重写；工具结果按固定顺序回灌
  4) alias / temp / top_p 保持一致（alias 变了就是另一个模型，缓存不通用）

用法：
    from bonsai_client import BonsaiClient
    c = LocalClient(system=SYSTEM, tools=TOOLS)      # 前缀冻结
    print(c.ask("上海天气?"))
    print(c.ask_tool("查一下 D:\\AI\\LLMlocal 下有哪些文件"))   # 自动处理 tool_calls
    c.report()                                        # 看缓存命中率
"""
import json
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_ALIAS = "lfm25-2.6b"          # llama-server --alias，见 agent-*.cmd
DEFAULT_TEMP = 0.1                 # LFM2.5 模型卡值；Bonsai 用 0.3~0.5


class LocalClient:
    def __init__(self, system, tools=None, url=DEFAULT_URL, alias=DEFAULT_ALIAS,
                 temperature=DEFAULT_TEMP, max_tokens=512, timeout=900,
                 top_k=None, seed=42, prefill_tps=47.9):
        self.url = url
        self.alias = alias
        self.temperature = temperature
        self.top_k = top_k
        self.seed = seed
        self.prefill_tps = prefill_tps   # 满缓存时省下的秒数 = 复用token / 这个速度
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 前缀：system 之后不再改动
        self._prefix = [{"role": "system", "content": system}]
        self.tools = tools or []
        self.stats = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0,
                      "fresh_tokens": 0, "prefill_time": 0.0, "gen_tokens": 0,
                      "gen_tps_sum": 0.0}
        self.last = None

    # ---------- 内部 ----------
    def _post(self, messages, tools=None, extra=None):
        body = {
            "model": self.alias,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,      # 固定 seed -> 同样输入同样输出，便于复现
        }
        if self.top_k is not None:
            body["top_k"] = self.top_k
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if extra:
            body.update(extra)
        req = urllib.request.Request(
            self.url, json.dumps(body, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
        wall = time.time() - t0

        u = d.get("usage", {}) or {}
        tm = d.get("timings", {}) or {}
        # 这个版本里最有用的三个字段（比 usage 更精确）:
        #   cache_n   = 直接复用的前缀 token 数
        #   prompt_n  = 本次真正新算的 prompt token 数
        #   prompt_ms = 只算新 token 的耗时 -> 满缓存时接近 0
        cached = tm.get("cache_n") or (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ptok = u.get("prompt_tokens", 0) or (tm.get("prompt_n", 0) + tm.get("cache_n", 0))
        ttft = tm.get("prompt_ms", 0) / 1000.0
        gen_tok = tm.get("predicted_n", u.get("completion_tokens", 0))
        gen_tps = tm.get("predicted_per_second") or 0.0

        s = self.stats
        s["calls"] += 1
        s["prompt_tokens"] += ptok
        s["cached_tokens"] += cached
        s["fresh_tokens"] += tm.get("prompt_n", ptok - cached)
        s["prefill_time"] += ttft
        s["gen_tokens"] += gen_tok
        s["gen_tps_sum"] += gen_tps

        self.last = {
            "message": d["choices"][0]["message"],
            "finish": d["choices"][0].get("finish_reason"),
            "prompt_tokens": ptok,
            "fresh_tokens": tm.get("prompt_n", ptok - cached),
            "cached_tokens": cached,
            "cache_hit_pct": (cached / ptok * 100.0) if ptok else 0.0,
            "ttft_s": ttft,
            "gen_tps": gen_tps,
            "wall_s": wall,
        }
        return self.last

    # ---------- 对外 ----------
    def ask(self, text, append_user=True):
        """普通一轮对话。只追加，历史不重写 -> 前缀全程可复用。"""
        if append_user:
            self._prefix.append({"role": "user", "content": text})
        r = self._post(self._prefix, self.tools)
        self._prefix.append({"role": "assistant", "content": r["message"].get("content") or ""})
        return r

    def ask_tool(self, text, tool_impl, max_rounds=4):
        """
        带工具的一轮：模型要调工具 -> 你执行 -> 结果回灌 -> 再问，直到它给最终答案。
        tool_impl: dict[str, callable]，名字与 tools 里的 function.name 对应。
            回调收到的是**已解析的 dict**（自动 json.loads 工具参数）。
        """
        self._prefix.append({"role": "user", "content": text})
        for _ in range(max_rounds):
            r = self._post(self._prefix, self.tools)
            m = r["message"]
            tcs = m.get("tool_calls")
            if not tcs:
                self._prefix.append({"role": "assistant", "content": m.get("content") or ""})
                return r
            # assistant 侧：保留 tool_calls 原文（保证后续轮次前缀稳定）
            self._prefix.append({"role": "assistant", "content": m.get("content") or "",
                                 "tool_calls": [{
                                     "id": t["id"], "type": "function",
                                     "function": {"name": t["function"]["name"],
                                                  "arguments": t["function"]["arguments"]}}
                                     for t in tcs]})
            # 工具结果按模型给出的顺序回灌（顺序稳定 -> 可缓存）
            results = []
            for t in tcs:
                name = t["function"]["name"]
                fn = tool_impl.get(name)
                args = t["function"].get("arguments") or "{}"
                if isinstance(args, str):           # 多数后端返回 JSON 字符串
                    try:
                        args = json.loads(args)
                    except Exception:                # 坏了就原样交给实现自己处理
                        args = {"_raw": args}
                try:
                    out = json.dumps(fn(args) if fn else
                                     {"error": "tool not implemented: %s" % name},
                                     ensure_ascii=False)
                except Exception as e:                     # noqa: BLE001
                    out = json.dumps({"error": str(e)}, ensure_ascii=False)
                results.append({"role": "tool", "tool_call_id": t["id"], "name": name,
                                "content": out})
            self._prefix.extend(results)
        return self.last

    def report(self):
        s = self.stats
        if not s["calls"]:
            return "(还没有调用)"
        n = s["calls"]
        hit = s["cached_tokens"] / s["prompt_tokens"] * 100.0 if s["prompt_tokens"] else 0.0
        tps = s["gen_tps_sum"] / n
        saved = s["cached_tokens"] / 47.9      # 47.9 tok/s 实测 prefill 速度
        return ("调用 %d 次 | prompt %d tok (复用 %d = %.1f%%, 新算 %d) | 平均 prefill %.2f s "
                "| 生成 %.1f t/s | 缓存省下 ≈ %.0f s"
                % (n, s["prompt_tokens"], s["cached_tokens"], hit, s["fresh_tokens"],
                   s["prefill_time"] / n, tps, saved))

    def history(self):
        return list(self._prefix)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r"D:\AI\LLMlocal")
    c = LocalClient(
        system="你是本地命令行 agent，回答用中文，简洁分点，不超过 5 行。",
        tools=[{"type": "function", "function": {
            "name": "list_dir",
            "description": "列出目录下的文件",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]}}}])
    c.ask("一句话说明什么是向量数据库")
    c.ask("我刚才问了你什么？一句话")
    print(c.report())


# 旧名保留，方便历史脚本/笔记继续跑
BonsaiClient = LocalClient
