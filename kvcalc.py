#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kvcalc.py — 读 GGUF 元数据，算 KV cache 占用 / 预估 decode 速度 / 上下文半衰点

    python kvcalc.py <model.gguf> [更多 gguf...]

为什么需要它：纯 CPU 上 decode 速度由内存带宽决定

    t/s  ≈  BW / ( W + L x KV_per_token )

  W            每 token 要读的权重字节（MoE 只读激活的 expert，不是整个文件）
  KV_per_token 架构决定，混合注意力架构只有部分层存 KV
  L            当前对话的实际 token 数（不是 -c 的上限！）

所以「窗口开多大不影响速度」和「长对话会变慢」两件事同时成立 ——
前者说的是 -c，后者说的是 L。这个脚本把 L_half = W / KV_per_token 算出来，
即「对话多长时 KV 读取开销等于权重读取开销」，超过这个点速度开始腰斩。

    python kvcalc.py models/*.gguf --ctx 32768
"""
import os
import struct
import sys

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
          5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    """返回 (metadata dict, tensors list of (name, dims))"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF file"

        def u(fmt):
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        def rd_str():
            return f.read(u("<Q")).decode("utf-8", "replace")

        def rd_val(t):
            if t == 8:
                return rd_str()
            if t == 9:
                et = u("<I")          # 元素类型只读一次
                n = u("<Q")
                return [rd_val(et) for _ in range(n)]
            return u(SCALAR[t])

        u("<I")                       # version
        n_tensors, n_kv = u("<Q"), u("<Q")
        meta = {}
        for _ in range(n_kv):
            k = rd_str()
            meta[k] = rd_val(u("<I"))

        tensors = []
        for _ in range(n_tensors):
            name = rd_str()
            nd = u("<I")
            dims = [u("<Q") for _ in range(nd)]
            u("<I")                   # ggml type
            u("<Q")                   # data offset
            tensors.append((name, dims))
        return meta, tensors


def analyse(path, ctx_list, bandwidth=17.0):
    meta, tensors = read_gguf(path)
    arch = meta.get("general.architecture", "?")
    name = os.path.basename(path)
    w_bytes = os.path.getsize(path)
    def g(key, default=None):
        v = meta.get(arch + "." + key)
        if v is None:
            v = meta.get(key)
        return default if v is None else v

    n_embd = g("embedding_length", 0) or 0
    n_head = g("attention.head_count", 0) or 0
    head_dim = g("attention.key_length", 0) or (n_embd // n_head if n_head else 0)
    v_dim = g("attention.value_length", 0) or head_dim
    n_layer = g("block_count", 0) or 0

    # ---- 哪些层真的存 KV ----
    # 混合注意力（lfm2 / qwen35）只有 full-attention 层存 KV，其余是
    # short-conv 或 linear-attention(SSM)，状态恒定、不随上下文增长。
    # 两个权威信号：
    #   qwen35.full_attention_interval = 每 N 层一个 full-attn 层
    #   lfm2*.attention.head_count_kv  = 逐层列表，conv 层写 0
    n_attn = None
    n_kv_head = None
    interval = g("full_attention_interval", 0) or 0
    if interval and n_layer:
        n_attn = len(range(0, n_layer, interval))
    raw_kv = g("attention.head_count_kv")
    kv_list = None
    if isinstance(raw_kv, list) and raw_kv:
        kv_list = raw_kv
        nz = [v for v in kv_list if v]
        if nz:
            n_attn = len(nz)
            n_kv_head = nz[0]
    if n_attn is None:
        n_attn = n_layer
    if n_kv_head is None:
        n_kv_head = g("attention.head_count_kv") or n_head
    hybrid = n_attn < n_layer

    if not (n_attn and n_kv_head and (head_dim + v_dim)):
        print("%-44s arch=%s  无法推出 KV 参数，跳过" % (name, arch))
        return

    # K 和 V 分开存，(head_dim + v_dim) 已经涵盖两者，不要再乘 2
    per_tok = n_attn * n_kv_head * (head_dim + v_dim) * 2      # f16 = 2 bytes

    print("=" * 76)
    print("%s   权重 %.2f GB" % (name, w_bytes / 1e9))
    print("  arch=%-8s 总层 %-3d 存KV层 %-3d%s" %
          (arch, n_layer, n_attn,
           "  <- 混合注意力，%d 层是 conv/SSM 不占 KV" % (n_layer - n_attn) if hybrid else ""))
    print("  head %s x head_kv %s x head_dim %s   原生窗口 %s" %
          (n_head, n_kv_head, head_dim, g("context_length", "?")))
    print("  KV = %.1f KiB/token (f16)      L_half = %.1fk tokens" %
          (per_tok / 1024, w_bytes / per_tok / 1000))
    print()
    print("  %-9s %8s %8s %8s %8s   %s" %
          ("对话长度", "KV占用", "KV/权重", "预估tg", "相对满速", "说明"))
    base = bandwidth / (w_bytes / 1e9)          # 空上下文时的带宽上限
    for L in ctx_list:
        kv_gib = per_tok * L / 1024 ** 3
        ratio = per_tok * L / w_bytes            # KV 读取 / 权重读取
        tps = base / (1 + ratio)
        note = ""
        if ratio > 1:
            note = "KV 读取已超过权重，瓶颈完全在 KV"
        elif ratio > 0.3:
            note = "明显掉速"
        print("  %-9s %7.2fG %7.2fx %7.1f %7.0f%%   %s" %
              ("%dk" % (L // 1000) if L >= 1000 else str(L),
               kv_gib, ratio, tps, 100.0 / (1 + ratio), note))
    print()


if __name__ == "__main__":
    argv = sys.argv[1:]
    ctx, bw = [32768], 17.0
    files = []
    i = 0
    while i < len(argv):
        if argv[i] == "--ctx":
            i += 1
            ctx = [int(x) for x in argv[i].split(",")]
        elif argv[i] == "--bw":
            i += 1
            bw = float(argv[i])
        else:
            files.append(argv[i])
        i += 1
    if not files:
        print(__doc__)
        sys.exit(1)
    for p in files:
        try:
            analyse(p, ctx, bw)
        except Exception as e:
            print("%-44s 读取失败: %s" % (os.path.basename(p), e))
