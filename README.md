# LLMlocal — 纯 CPU 本地大模型推理 / Agent 工具调用工作台

在**一台没有 GPU 的 Windows 机器**上，把 GGUF 模型跑成 OpenAI 兼容的本地 API，
专门用来给 agent 做工具调用（tool_calls）。所有性能数字都是在这台机器上实测的，
不是抄的规格表。

**硬件基线**：i5-8500（6 核 6 线程，AVX2，无 AVX512/VNNI）· 19.86 GiB 内存 · 无独显
（仅 UHD 630 iGPU）· DDR4 实测带宽 ~16 GB/s
**推理引擎**：默认 llama.cpp **PrismML fork b10743 / commit adfffbe41**
（必须用 `Bonsai-demo/bin/cpu/`，见「坑」一节）
**2.6B 专用引擎**：**ik_llama.cpp @ `1a88bc0`，本机自编译 AVX2**（`ik_llama-src/build/bin/`）。
它的 IQK FlashAttention CPU kernel 直接吃量化 KV，**预填充比旧引擎快 32~60%**，
`agent-lfm26.cmd` 已切过去，见 4.17。其余脚本仍用 b10743。

---

## 1. 三十秒上手

```cmd
:: 1. 双击一个启动脚本（保持窗口不关，关窗口 = 停服务）
agent-lfm25-8b.cmd          :: 推荐：工具调用 agent 用这个，端口 8082
agent-lfm26.cmd              :: 要写文件/改代码用这个，路径最准，端口 8080
agent-lfm12-fast.cmd         :: 只要快、不写文件用这个，端口 8080（与上面互斥）
agent-sakthai.cmd            :: 最快的底座（35.7 t/s），端口 8084，**工具调用走 8085 代理**

:: 2. 另开一个 cmd 验证
python tool_test.py 8082 lfm25-8b 0.1          :: 冒烟：能不能吐 tool_calls
python agent_loop.py 5 8082 lfm25-8b 0.1       :: 完整 5 轮 agent 循环 + 缓存统计
python fs_probe.py 8082 lfm25-8b 0.1 --clean   :: 真写一次文件（冒烟不验参数，见 4.14）

:: 3. 停服务
powershell -c "Stop-Process -Name llama-server -Force"
```

服务起来后：`http://127.0.0.1:<端口>/v1`（OpenAI 兼容），Web UI 在 `http://127.0.0.1:<端口>/`
（用 `127.0.0.1` 而不是 `localhost` —— 服务只绑 IPv4，浏览器可能先解析 `::1`）。

### 1.1 本机环境（所有实测数字的前提，换机器必须重测）

| 项 | 值 | 备注 |
|---|---|---|
| CPU | i5-8500 @ 3.0 GHz，**6 核 6 线程** | AVX2，**无** AVX512/VNNI —— i-quant 在这台机器上是陷阱（见 4.4） |
| 内存 | 19.86 GiB DDR4，实测带宽 ~16 GB/s | decode 速度的天花板 |
| 独显 | **无**。仅 UHD 630 iGPU（系统报 10298 MiB，实际可用 0） | `-ngl 0`，Vulkan 后端直接放弃 |
| 其他显卡 | AMD R7 200 Series / Todesk 虚拟显示器 | 同样不可用 |
| 磁盘 | C: 238.5 GiB（剩 53.6）· D: 931.2 GiB（剩 472.9） | 权重全部放 D: |
| OS | Windows 10 企业版 LTSC 19044（**代码页 936**） | 这一条决定了 `.cmd` 必须是纯 ASCII，见 6 |
| 引擎（默认） | llama.cpp **PrismML fork build 10743 / commit adfffbe41** | Clang 20.1.8，只用 `Bonsai-demo/bin/cpu/` |
| 引擎（2.6B 用） | **ik_llama.cpp `1a88bc0`，自编译 AVX2** | `ik_llama-src/build/bin/`，只有 2.6B 在用，见 4.17 |
| 引擎源码 | `llama-cpp-src` @ `81ef10e`（上游 ggml-org/llama.cpp） | 改 Q2_0 支持时对照用 |
| 子项目 | `Bonsai-demo` @ `9ef3205`（PrismML-Eng/Bonsai-demo） | 二进制来源 |
| 子项目 | `ik_llama-src` @ `1a88bc0`（ikawrakow/ik_llama.cpp） | 自编译，构建脚本见 4.17 |
| 工具链 | Python 3.14.5 · Git 2.55.0 · **MSYS2 gcc 16.2.0 + cmake 4.4.3 + ninja 1.13.2** | 打印中文需要 `PYTHONIOENCODING=utf-8`；编译器装在 `C:\msys64` |
| 空闲内存 | 实测 3.5–8.5 GiB 波动 | 所以每个启动脚本都现场跑 `llama-plan.ps1` 降级 |

**换机器时必须重测的三件事**：`llama-bench -p 2048`（**`-p 0` 的 tg 只在部分模型上可信**，见 6）、
`llama-plan.ps1 -Size lfm12 -Ctx 16384`（planner 有按模型编的峰值表）、
`fs_probe.py`（换模型或换 harness 后工具保真度可能完全变样）。
本仓库里没有一条性能数字是抄来的，全部来自上面这台机器。

---

## 2. 模型选型（实测，先看这个表）

| 启动脚本 | 模型 | 权重 | 端口 | 生成 | 预填充 | 工具调用 | 定位 |
|---|---|---|---|---|---|---|---|
| `agent-sakthai.cmd` | SakThai Context 0.5B Q4_K_M | 0.37 GiB | 8084 → **8085 代理** | **35.7 t/s** | 213 t/s | ✅ 4/4（经代理） | ⚠️ **最快**，但只能只读/问答（见 4.16） |
| `agent-minicpm5-1b.cmd` | MiniCPM5-1B Q4_K_M | 0.64 GiB | 8081 | **26.8 t/s** | 215 t/s | ❌ 拒绝 | 起草/摘要/长上下文预热 |
| `agent-lfm12-fast.cmd` | LFM2.5-1.2B-Instruct Q4_K_M | 0.68 GiB | 8080 | **21.7 t/s** | 163 t/s | ✅ 3/3 | ⭐ **最快的 agent 底座**（2k→16k 零衰减） |
| `agent-lfm26.cmd` | LFM2.5-2.6B Q4_K_M | 1.56 GiB | 8080 | 8.6 t/s | **88.5 t/s** | ✅ 3/3 | ⭐ **agent 要写文件就用它**（路径不打错）；已换 ik_llama.cpp 引擎，见 4.17 |
| `agent-lfm25-8b.cmd` | LFM2.5-8B-A1B Q4_K_M | 4.96 GiB | 8082 | 13.5 t/s | 79.7 t/s | ✅ | ⭐ **真实工具循环首选** |
| `agent-bonsai17b.cmd` | Ternary-Bonsai-1.7B-PQ2_0 | 0.43 GiB | 8080 | **34 t/s** | 47.9 t/s | ✅ | 纯聊天最快，32k 原生窗口 |
| `agent-ornith9b.cmd` | Ornith-1.5-9B Q4_K_M | 5.50 GiB | 8080 | 2.9 t/s | 25.5 t/s | ✅ | ⭐ **写代码最强**，262k 上下文 |
| `agent-qwen7b.cmd` | Qwen2.5-Coder-7B Q4_K_M | 4.36 GiB | 8080 | 3.9 t/s | 26.4 t/s | ❌ 退化成纯文本 | 快 11% 但工具调用废了 |
| `agent-bonsai8b.cmd` | Ternary-Bonsai-8B（模型未下载） | — | 8080 | 8.2 t/s | — | — | 已被 8B-A1B 取代 |

> SakThai 0.5B 的 35.7 t/s 是**短上下文**实测值；它的 KV 是 24 层全存（12 KiB/token），
> 8k 掉到 17.9、16k 掉到 9.7，比 LFM2.5-1.2B（2k→16k 零衰减）衰减快得多。
> 「生成」列一律是短上下文的数字，长对话请查 4.6 的曲线。
> 2.6B 那一行的两个数字是 `llama-bench -t 4 -p 2048` 的实测（与 4.17 表同源），
> 换引擎后跟旧表里 10.3 / 65.5 那组不可直接比。

**真实 agent 前缀端到端**（`agent_loop.py --warm`，含 system + 3 工具）：

| 模型 | 耗时 | 综合速度 | 相对 |
|---|---|---|---|
| MiniCPM5-1B | 6.3 s | 136.2 t/s | 1.0×（但不会调工具） |
| LFM2.5-8B-A1B | 21.5 s | 35.4 t/s | 3.4× |
| Ornith-1.5-9B Q4_K_M | 43.7 s | 22.0 t/s | 6.9× |
| Ornith-1.5-9B IQ4_XS | 77.0 s | 12.4 t/s | 12.2× |

### 该选哪个

- **只要快 + 要调工具** → `agent-lfm12-fast.cmd`（端口 8080）。decode 21.7 t/s 且
  **2k→16k 零衰减**，16k 上下文只占 1.2 GiB，5 轮循环 16.7 s。
- **只要快** → `agent-sakthai.cmd`（**端口 8085 代理**，35.7 t/s，397 MB）。
  快 1.6 倍，工具调用经 `sakfix.py` 归一化后 `agent_loop.py` 5/5 轮全对；
  但 `fs_probe.py` 明确轮仍 FAIL，会凭空编 `/home/user/x.txt`。只读 / 分类 / 摘要，
  见 4.16。
- **工具调用 agent，且需要 8B 级知识** → `agent-lfm25-8b.cmd`。8.47B 总参但每 token
  只激活 1.5B，速度像 1.5B，知识像 8B。代价是 decode 慢 1.6×（21.7 → 13.8 t/s）。
- **agent 要写真代码** → `agent-ornith9b.cmd`（SWE-bench Verified 70.6），
  代价是慢 7 倍。
- **纯聊天 / 最快交互** → `agent-bonsai17b.cmd`（34 t/s，但 L>4k 就腰斩）。
- **草稿、摘要** → `agent-minicpm5-1b.cmd`（26.8 t/s，但不会调工具）。

---

## 3. 目录结构

LLMlocal/
├─ README.md                    ← 本文件（总入口）
├─ README-lfm25.md              ← LFM2.5 家族详解（工具调用、采样参数、thinking 坑）
├─ README-qwen.md               ← Qwen/Ornith 实测报告（量化对比、KV、下载 BUG）
├─ README-bonsai.md             ← Bonsai 1.7B 详解（32k 窗口、前缀缓存、调参）
│
├─ agent-*.cmd                  ← 【启动 server】给 agent 用的 OpenAI 兼容 API
├─ start-*.cmd                  ← 【交互聊天】llama-cli 自己聊天的终端
│
├─ llama-plan.ps1               ← 资源规划器：按实时空闲内存自动降级 KV/上下文
│
├─ llm_client.py                ← 前缀缓存友好的 OpenAI 客户端封装（LocalClient）
├─ agent_loop.py                ← 5 轮真实 agent 循环测试
├─ tool_test.py                 ← 3 用例 tool_calls 冒烟（退出码 0/1）
├─ tool_stress.py               ← N 工具压力测试，定位「工具一多就选错」的拐点
├─ fs_probe.py                  ← 写文件能力探针（沙箱真落盘，退出码 0/1）
├─ bench_decode.py              ← decode 速度 vs 上下文长度曲线（旧，差分法）
├─ bench_ctx2.py                ← 同上，但直接读 server 端 timings（**优先用这个**）
├─ ctxtune.sh                   ← 长上下文下批量对比 KV 量化 / 线程 / fa / mlock
├─ cache_test.py / ttft_test.py ← 前缀复用率 / 首字延迟
├─ kvcalc.py                    ← 读 GGUF 元数据算 KV cache 占用
├─ ctxsweep.sh / tbench.sh      ← decode vs 上下文 / vs 线程数 批量测试
├─ dl.sh                        ← GGUF 下载器（魔搭→hf-mirror→HF，多路分片 + 校验）
│
├─ sakfix.py                    ← **SakThai 归一化代理**（8085→8084）：5 种野格式→tool_calls、截废话、截退化重复、修历史 tool_calls（见 4.16）
├─ sak_unit.py                  ← sakfix.py 归一化逻辑单测（26 条，秒级，不碰模型）
├─ sak_diag.py                  ← SakThai 原始输出格式诊断（把野格式逐个量化）
├─ sak_tokcfg.json             ← 模型卡 tokenizer_config 存档（对照训练时的 tool 格式）
├─ sak_template.jinja           ← SakThai 0.5B 的严格版 chat template（挂不挂都行，见 4.16）
├─ sakstart.sh                  ← SakThai 一次性测试启动器
├─ sak_qa.py                    ← 中文问答/代码分析质量（不给工具）
├─ sak_eval.py                  ← 工具调用 + 中文分析 4 用例，带 stop 对照
├─ sak_oneshot.py               ← one-shot 能否把格式合规率救回来
├─ sak_forced.py                ← tool_choice 指定函数时的 grammar 约束路径
├─ raw_probe.py                 ← 看模型吐的原始文本（绕过 tool_call 解析器）
│
├─ models/                      ← 所有 GGUF（35 GB，统一平铺，不分子目录；**不入库**）
├─ Bonsai-demo/                 ← 上游 repo（PrismML-Eng/Bonsai-demo，**不入库**）
│  └─ bin/cpu/                  ← ⭐ 实际使用的 llama.cpp 二进制（b10743）
├─ bin-vulkan/                  ← Vulkan 构建（故意不放进 Bonsai-demo/bin，避免误选）
├─ llama-cpp-src/               ← llama.cpp 源码（ggml-org/llama.cpp，参考用）
├─ ik_llama-src/               ← ik_llama.cpp 源码 + 自编译产物（**不入库**）
│  └─ build/bin/                ← ⭐ 2.6B 实际使用的 ik 引擎（AVX2，见 4.17）
├─ ATLAS-engine/                ← ATLAS TQ1.0 三值推理引擎（独立 C/C++ 项目）
└─ .gitignore                   ← 上面带「不入库」的都是排除项

---

## 4. 核心机制

### 4.1 资源规划 `llama-plan.ps1`

每个启动脚本启动前都会调它问一句「现在跑得起什么」，按**实时**空闲内存决策，
不会超过 `总内存 - 10% 余量 - 已用量`，线程数硬上限 5/6 核：

```
降级顺序（质量损失最小优先）：
  f16 → q8_0 → q4_0（KV 量化）
  131072 → 65536 → 32768 → 16384 → 8192（上下文减半）
```

内置 10 张实测峰值表（`-Size`）：`8b / 17b / lfm12 / lfm25 / lfm258b / mcpm5 /
q7b / q30b / ornith9b / ornith35b`。未测量的组合会按「权重恒定 + KV 线性」插值并向上取整。

```powershell
# 手工查一下能跑多少
powershell -NoProfile -ExecutionPolicy Bypass -File llama-plan.ps1 -Size lfm258b -Ctx 32768 -Ctk f16 -MaxRam 9
powershell ... -Only NOTE     # 只打印一句人话说明（脚本用这个）
```

> ⚠️ 表里记的是 **WorkingSet**，不是 private commit。mmap 的权重落在文件缓存里，
> Windows 会在压力下回收。读内存占用要同时看 `PrivateWS`（真占）和 `WorkingSet`（含 mmap）。

### 4.2 前缀缓存（CPU 上最大的免费加速）

llama.cpp 按 **token 序列**找最长公共前缀。实测 2275 token 的 system prompt：
冷启动 TTFT **82.6 s** → 命中后 **0.67 s**（复用 99.2%）。6 轮 agent 循环 TTFT 1.5–2.5 s，
复用率 90–94%。**客户端姿势比模型参数更影响体感**，四条铁律：

1. **system prompt + tools 放最前，逐字节不变** —— 千万别插「当前时间 / 会话 ID /
   随机数」，一次就废掉整个前缀。
2. **tools 数组顺序与数量不变** —— 工具 schema 也是 prompt 的一部分。
3. **历史只追加、不重写**；工具结果按固定顺序回灌。
4. **alias / temp / top_k 保持一致**（alias 变了就是另一个模型，缓存不通用）。

```python
import sys; sys.path.insert(0, r"D:\AI\LLMlocal")
from llm_client import LocalClient

c = LocalClient(system=SYSTEM, tools=TOOLS)     # 前缀冻结，之后别动
c.ask("一句话说明什么是向量数据库")
c.ask_tool("列出 models 目录下的文件", {"list_dir": my_func})
print(c.report())   # 调用数 / 复用率 / 平均 prefill / 缓存省下的秒数
```

观察点：响应 `timings` 里 `cache_n` = 复用前缀 token，`prompt_n` = 新算的，
`prompt_ms` 只算新 token（满缓存时接近 0）。

配套的 server 端开关：`--cache-ram N --ctx-checkpoints 64 --checkpoint-min-step 512
--cache-reuse 64 --cache-idle-slots`。`--checkpoint-min-step` 默认 8192 对 agent 来说
太大，短 prompt 建不了检查点，必须调小。**这些参数不占内存**（是上限不是预留），
降低它们省不出 RAM，别折腾。

### 4.3 内存带宽就是天花板

DDR4 实测 ~16 GB/s，decode 速度 ≈ 权重字节数 ÷ 16 GB/s：

| 模型 | 权重 | t/s | 权重 × 速度 |
|---|---|---|---|
| Ternary-Bonsai-1.7B PQ2_0 | 0.44 GB | 36.25 | 15.8 GB/s |
| LFM2.5-1.2B Q4_K_M | 0.69 GB | 23.87 | 16.5 GB/s |
| LFM2.5-2.6B Q4_K_M | 1.55 GB | 9.94 | 15.3 GB/s |
| MiniCPM5-1B Q4_K_M | 0.64 GB | 26.76 | 17.1 GB/s |

**所以：加线程没用**（实测 t=3/4/5 持平，t=6 反而更慢），想快只能减权重 ——
换小模型，或换更激进的量化。

### 4.4 量化选择：i-quant 是 AVX2 的陷阱

**x86 上 i-quant（IQ*）没有 batched GEMM 通道，每个权重要按 batch 的每个 token
做一次 LUT 解码。** 同一权重上实测：

| 量化 | prefill (Ornith-9B) |
|---|---|
| IQ4_XS (4.25 bpw) | 13.7 t/s |
| **Q4_K_M (4.82 bpw)** | **25.5 t/s（+87%）** |

K-quant 走真正的 GEMM 路径。这台是纯 AVX2 的 Coffee Lake（无 AVX512、无 VNNI），
差距尤其大。**结论：优先 Q4_K_M，别为省 0.6 GB 换 IQ*。**

### 4.5 KV cache 量化：分模型看，不是全有全无

| 模型 | 建议 KV | 理由 |
|---|---|---|
| LFM2.5-8B-A1B / MiniCPM5-1B | **f16** | q4_0 让 prefill 慢 2.0–3.8×，只换来 decode +8~53%，不划算（agent 等的是冷 prefill） |
| Ternary-Bonsai-1.7B | **q4_0** | prefill 快 2×、生成快 5×、内存减半 |
| Ornith-1.5-9B | **q4_0** | 256k×q4_0 = 2.0 GiB，与 128k×q8_0 同价，但上下文翻倍且 decode 更快 |

原因：量化 KV 会让 llama.cpp 在 attention 里逐次反量化 K/V，正好落在 O(n²) 的
prefill 路径上；而权重（0.6–5 GiB）在现实上下文长度下仍主导 decode 的读取量。
**只 KV 极贵（注意力层少）的模型才划算。**

### 4.6 上下文长度：`-c` 不是对话长度

`llama-bench tg64` 实测 decode **对 `-c` 不敏感**（512 → 34.8 / 2k → 34.7 / 4k → 34.9 /
8k → 34.3 t/s），因为 decode 只遍历**实际存在**的 KV。
真正拖慢速度的是**当前对话有多长**：

| KV 实际长度 | LFM2.5-8B-A1B decode |
|---|---|
| 2.0k | 11.5 t/s |
| 7.2k | 7.6 t/s |
| 13.9k | 5.1 t/s |
| 17.8k | 3.1 t/s |

**缩小 `-c` 不会让生成变快**，只会让墙来得更早。想快就保持对话短 ——
`--context-shift` 会帮你滑动。注意 Ornith（混合线性注意力）**不支持 KV shifting**，
`--cache-reuse` / `--context-shift` 会被静默禁用，262144 是硬上限，满了只能开新会话。

### 4.7 带宽公式：先算再选模型

`kvcalc.py` 直接从 GGUF metadata 算，不用猜：

```
t/s  ≈  BW / ( W + L x kv )

  W     权重字节（MoE 每 token 只读激活的 expert，不是整个文件）
  kv    KV bytes/token = 存KV层数 x head_kv x (head_dim_k + head_dim_v) x 2
  L     当前对话实际 token 数 —— 不是 -c 上限
  BW    DDR4 实测 ~16 GB/s

L_half = W / kv   ← 对话到这个长度，KV 读取开销等于权重读取，速度腰斩
```

混合注意力架构（`lfm2` / `qwen35`）只有少数层存 KV，其余是 short-conv 或
linear-attention（SSM 状态恒定、不随上下文增长），所以 `kv` 便宜 2–9 倍。
GGUF 里的权威信号：`qwen35.full_attention_interval`、或 `attention.head_count_kv`
是个逐层列表（conv 层写 0）。

**本机实测 kv 成本**（`python kvcalc.py models/*.gguf`）：

| 模型 | 权重 | 存KV层 | kv/token | L_half |
|---|---|---|---|---|
| **LFM2.5-8B-A1B** | 5.32 GB | 6/24 | **12 KiB** | **433k** |
| LFM2.5-1.2B | 0.73 GB | 6/16 | 12 KiB | 60k |
| MiniCPM5-1B | 0.69 GB | 24/24 | 24 KiB | 28k |
| Ornith-1.5-9B | 5.91 GB | 8/32 | 32 KiB | 180k |
| Qwen2.5-Coder-7B | 4.68 GB | 28/28 | 56 KiB | 82k |
| **Ternary-Bonsai-1.7B** | 0.46 GB | 28/28 | **112 KiB** | **4.0k** ⚠️ |

### 4.8 结论：选 LFM2.5-8B-A1B + 16k

```cmd
set "CTX=16384"     :: 配合 --context-shift 把当前对话钉在 16k 内
set "CTK=f16"       :: 保持 f16！q4_0 让 prefill 慢 2×，agent 等的是冷启动
set "THREADS=4"     :: t=4 最快，t=6 反而慢 55%（内存总线争抢）
set "TB=6"          :: prefill 单独用 6 线程
set "UB=2048"       :: MoE 需大 ubatch 喂饱 32 个 expert（92 vs 78 t/s）
```

三条理由：

1. **kv/token 最低（12 KiB）** —— 16k 对话下 KV 读取仅占权重 4%，decode 几乎不因上下文掉速
2. **MoE 每 token 只激活 4/32 expert**，5.32 GB 权重但只读 ~1/8，比 Ornith 9B 快 4.3×
3. **支持 `--context-shift` + `--cache-reuse`**（混合注意力里少见的），长对话自动滑动

### 4.9 降低 `-c` 为什么会变快（反直觉）

`-c` 本身**不影响速度** —— decode 按需读 KV，`llama-bench tg64` 实测 512→34.8、
8k→34.3 t/s 恒定。但**配合 `--context-shift` 就不同了**：

```
-c = 滑动窗口的天花板
     ↓
天花板越小 → 滑动越早触发 → 当前对话越短
     ↓
当前对话越短 → 每 token 要扫的 KV 越少 → 越快
```

实测（`-c 32768 / f16 / -t 4`）：

| 当前 KV 长度 | decode | prefill |
|---|---|---|
| 2.0k | 11.5 t/s | 86.0 t/s |
| 7.2k | 7.6 t/s | 86.7 t/s |
| 13.9k | 5.1 t/s | 80.7 t/s |
| 17.8k | 3.1 t/s | 60.3 t/s |

所以 `CTX` 从 32768 → 16384，实际效果是**把墙从 ~3 t/s 挪到 ~7 t/s**。
prefill 是 O(n²)（86 → 60 t/s），更小的上限也限制了单次请求的最长阻塞时间。

代价可以忽略：kv = 12 KiB/token，16k 只有 0.19 GiB，而实测真实 agent 前缀
只有 754 token —— **20 倍余量，正常工具循环根本碰不到墙**。

需要一次性读超长文件时再临时调回：`set CTX=131072`（需 YaRN，PrivateWS 5.23 GiB）。

### 4.10 IQ2 vs Q4_K_M：实测不是直觉

同一台机器、同一组参数（`-t 4 -ub 2048 -p 2048 -n 32 -r 2`，`llama-bench`）：

| 量化 | 大小 | bpw | **tg32** | pp2048 |
|---|---|---|---|---|
| `LFM2.5-8B-A1B-UD-IQ2_M` | 2.56 GiB | 2.7 | **13.84** ± 1.95 | 25.98 |
| `LFM2.5-8B-A1B-UD-Q4_K_M` | 4.95 GiB | 4.8 | 11.84 ± 0.38 | **73.69** |

**权重砍掉 48%，decode 只快 17%。** 原因：MoE 每 token 只读激活的 4/32 expert，
砍掉的那部分权重**本来就不在读取路径上**。而 i-quant 没有 batched GEMM，
省了带宽却赔了计算 —— 这个交易对 decode 不划算，对 prefill 更是亏爆（2.8×）。

**选型结论**：

| 场景 | 选择 | 理由 |
|---|---|---|
| **工具调用 agent** | **Q4_K_M** | 每轮都要冷 prefill 长前缀，73.7 vs 26.0 决定体感 |
| 纯聊天、单轮问答 | IQ2_M | decode +17%，且省 2.4 GB 内存 |
| 内存极度紧张 | IQ2_M | WorkingSet 6.3 vs 8.4 GiB |

> ⚠️ 内存不够时的正确解法**不是**降量化，而是修 planner —— 见 4.11。

### 4.11 `-CountMmap`：修掉 planner 的过度降级

`llama-plan.ps1` 的 `$PEAK_*` 表记的是 **WorkingSet**，它把 mmap 的权重页算进去了。
但那些是**文件后备、可回收**的页面。实测同一 server 只改 `-c`：

| ctx\|kv | WorkingSet | PrivateWS | 差值 |
|---|---|---|---|
| 32768\|f16 | 8.48 GiB | 3.72 GiB | **4.76 GiB** |
| 131072\|f16 | 9.71 GiB | 5.23 GiB | **4.76 GiB** |

差值恒定 = 4.95 GiB 的 mmap 权重，**不随 `-c` 或 KV 类型变化**（KV 是预分配的私有内存）。

**Bug 后果**：用 16384/f16 请求时，planner 认为需要 8.1 GiB 而只允许 6 GiB，
于是降级到 **8192 + q4_0** —— 而 q4_0 KV 恰好是 agent 最差的选择（prefill 慢 2×）。

```powershell
# 修好后：8.4 GiB WS - 4.76 GiB 可回收 = 3.6 GiB 真实占用，通过
powershell ... -Size lfm258b -Ctx 16384 -Ctk f16 -CountMmap
```

`agent-lfm25-8b.cmd` 已默认带上此开关。1.2B / 2.6B 本来就能通过（EST 1.2 / 2.1），
未改动。

### 4.12 Qwen3-1.7B：为什么“1 GB + 30 t/s”不能用

`models/Qwen3-1.7B-Q4_K_M.gguf` 看上去很美（1.03 GiB，宣称 30 t/s）。实测不然：

| 上下文 | tg32 |
|---|---|
| 2048 | **15.19** ± 0.72 |
| 8192 | 14.02 |
| 16384 | **跑不完**（两次中断） |

**宣称值 30 t/s 在这台机器上复现不出来，实测只有一半。** 根因是 KV 架构，
不是 CPU：

| | Qwen3-1.7B | LFM2.5-1.2B |
|---|---|---|
| 架构 | `qwen3`，28 层**全部** full-attn | `lfm2`，16 层只有 6 层是真注意力 |
| KV/token | **224 KiB** | **12 KiB** |
| L_half | **4.0k** | 60k |
| 16k 时 KV | 3.5 GiB（prefill O(n²) 直接跑不完） | 0.19 GiB |

它把“权重 1 GB”当成了“占用小”，但 **224 KiB/token 的 KV 才是真正的杀手** ——
16k 上下文要 3.5 GiB KV，比它自己的权重还大 3.4 倍。

另外它是**普通 instruct 模型，没有 agentic 后训练**，工具调用保真度远不如
LFM2.5 家族（后者原生返回 OpenAI `tool_calls`，已验证 3/3）。

> 💡 选型口诀：**看 KV/token，不看权重大小。** 权重决定短对话速度，
> KV/token 决定长对话能不能跑。

### 4.13 `agent-lfm12-fast.cmd`：实测调优配置

基于上表的实测数据新建的专用脚本（端口 8080，与 `agent-lfm26.cmd` 互斥）。`llama-bench -t 4 -ub 512 -n 32`：

| 上下文 | tg32 | pp |
|---|---|---|
| 2048 | 21.69 | 163.2 |
| 8192 | **21.96** | - |
| 16384 | **22.12** | 112.5 |

**2k → 16k 零衰减**（21.69 → 22.12），这就是 12 KiB/token 的直接体现。

参数选择依据：

| 参数 | 值 | 理由 |
|---|---|---|
| `CTX` | 16384 | 实测平坦区；agent 真实前缀 754 token，20× 余量 |
| `CTK` | **f16** | 量化 KV 让 prefill 慢 2×（70.4 → 36.0），agent 每轮等的就是冷 prefill |
| `-t` | 4 | `-t 6` 实测更慢（Qwen3-1.7B：15.19 → 9.29），内存总线争抢 |
| `-tb` | 6 | prefill 单独吃满核，与 `-t` 独立 |
| `-ub` | 512 | **与 8B 相反**：dense 模型 ubatch 是 wash，512 实测最快（MoE 需要 2048） |
| `--temp/--top-k` | 0.1 / 50 | 模型卡值，改高会让 tool_call JSON 崩 |
| `--jinja` | **必开** | 加载 GGUF chat template；不开则 1.2B 一个 `tool_calls` 都发不出来 |

**端到端验证**（`tool_test.py` + `agent_loop.py`，2026-10-12）：

```
tool_test.py   -> 3/3 通过，exit=0
agent_loop.py  -> 5 轮 16.7 s | 21.5 t/s | 前缀复用 84.8% | 收尾作答 4/5
```

⚠️ **已知短板**：1.2B 偶尔会把路径打错（实测把 `D:/AI/LLMLocal/models`
写成了 `D:/AI/LLMLocal/models` 的错拼变体）。**agent 要执行文件路径时，
必须在客户端校验每条路径**，或改用 `agent-lfm26.cmd`（2.6B，端口 8080）。

### 4.14 工具能返回 ≠ 能干活：`fs_probe.py` 实测

`tool_test.py` 对 1.2B 报 3/3 通过，但把它接进 **pi** 写文件时全军覆没。
翻 pi 的会话 jsonl 还原出来的实况：

| 轮 | 用户说 | 模型行为 | 结果 |
|---|---|---|---|
| 1 | 新建一个1.txt 写入今天的天气 | 没调，反问「请提供完整路径和内容」 | — |
| 2 | 写入当前根目录 | `write(path=".")` | `EEXIST`（`.` 是目录） |
| 3 | 你帮我执行 | `write(path="D:/user/Desktop")` | `EEXIST` |
| 4 | 创建文件 | `edit(path="D:/user/Desktop")` | `EISDIR` |
| 5 | — | 文本：「由于当前无法直接操作文件系统…」 | 编借口 |

**文件名从头到尾没出现在任何一次 `path` 里。** 连续 3 次工具报错后，1.2B
不修参数而是开始编理由（甚至说出「路径里包含了 open 和 write」这种胡话）。
教训有两条：

1. **冒烟测试只验「选对工具名」，不验参数、不执行。** 所以另写了 `fs_probe.py`：
   照抄 pi 的 schema（小写 `read/write/edit/bash` + `path`），在沙箱里**真的落盘**。
2. **模型的「工具一多就选错」是规模效应**，工具越少越稳（见 `tool_stress.py`）。

同一天用 2.6B（`agent-lfm26.cmd`，10.3 t/s）跑同一个探针：

```
轮1 弱  新建一个1.txt 写入今天的天气   -> bash("date")          拿天气
轮2 弱  写入当前根目录                 -> write("D:\...\probe-sandbox\1.txt",
                                              content="今天的天气是：…晴 25C")
                                         真落盘 44 bytes
轮3 弱  创建文件                       -> read(同一个 1.txt)     读回来核对
轮4 明确 给绝对目录 + 文件名           -> write("D:\...\notes.txt")
                                         真落盘 77 bytes        PASS, exit 0
```

**注意轮 2。** 「写入当前根目录」这句话把 1.2B 彻底打死（它把目录名当成了
`path`），而 2.6B **记住了轮 1 里的文件名 `1.txt`**，把它解析成沙箱下的完整路径，
并真的把内容写了进去 —— 跨轮状态和路径拼接都是对的。

结论：**2.6B 拿到路径就能写对，1.2B 就算给了完整路径也大概率丢掉文件名。**
文件操作类任务（写代码、跑脚本、改配置）直接用 2.6B 或 8B-A1B，1.2B 只做只读和问答。

另外 2.6B 是 thinking 模型，`max_tokens` 必须给够：**600 会整轮
`finish_reason=length` + `content=''`**（看起来像不会调工具），探针默认 `AGENT_MAXTOK=1400`。



- **Ternary-Bonsai-1.7B**：112 KiB/token 极高（28 层全注意力），**L>4k 速度就腰斩**，
  3.2k 时从 34 → 15 t/s。只适合短对话纯聊天
- **Ornith-1.5-9B**：kv 32 KiB 尚可，但 5.91 GB 权重吃满带宽（tg 2.9 t/s），
  且不支持 context-shift
- **MiniCPM5-1B**：最快但**不会调工具**，且 24 层全注意力在 32k 时 KV 读取超过权重本身

> ⚠️ **L_half 是带宽模型的乐观估计**。实测衰减比公式更快，因为 prefill 的 O(n²)
> attention 和线程同步开销不在这条公式里。L=17.8k 时实测已掉到 3.1 t/s
> （公式估 10 t/s）。**实际对话控制在 8k 以内**，靠 `--context-shift` 兜底。

---

### 4.15 LFM2.5-2.6B 还能不能再快：能做的和不能做的

问题：decode 只有 10.3 t/s，对话长了掉到 6 t/s。**先算天花板**：
1.55 GiB 权重 ÷ 15.3 GB/s = 9.94 t/s —— 实测 9.18~10.1，**已经在带宽墙上**，
所以任何不减少「每 token 要读的字节数」的旋钮都动不了它。
（本节全部数字是 b10743 引擎下的；2.6B 现在跑 ik_llama.cpp，预填充已从 65.5
提到 88.5 t/s，decode 仍是 8.6 —— 天花板结论不变，见 4.17。）

**内存锁定：+6%，值得加**（`llama-bench -p 0 -n 96 -r 2`，t=4）：

| 加载方式 | tg96 | 差 |
|---|---|---|
| `mmap`（默认） | 9.18 ± 0.19 | — |
| `-lm mlock` | **9.79 ± 0.26** | +6.7% |
| `-lm none`（不用 mmap） | 9.69 ± 0.07 | +5.5% |
| `-lm mmap+mlock` | 9.64 ± 0.39 | +5.0% |
| `-fa off` | 9.28 ± 0.11 | +1% |
| `-ctk q8_0 -ctv q8_0` | 9.66 ± 0.04 | +5%（KV≈0 时没有意义） |

> 📌 本节更正了第 6 节的老条目「mlock 比 mmap 慢 30%」—— 在这套配置（`-ngl 0
> -fa on -t 4`，KV≈0）下测不出来，`mlock` 反而快一点。差异可能来自当时的
> 内存压力状态。**要稳就两个都开**（`-lm mmap+mlock`）。

**长上下文掉速的真实原因：掉的是延迟，不是带宽。**
用 `bench_ctx2.py`（读 server 端 timings，不靠墙钟相减）实测 `-c 16384 / f16 / t 4`：

| KV 实际长度 | prefill | decode | 带宽公式预测 |
|---|---|---|---|
| 0.6k | 36.1 t/s | **10.1 t/s** | 9.9 |
| 1.9k | 32.6 | 9.6 | 9.6 |
| 3.6k | 22.9 | 8.7 | 9.1 |
| 7.0k | 12.8 | **6.2** | 8.5 |
| 10.4k | 9.0 | 4.9 | 8.1 |
| 13.8k | 6.9 | **4.1** | 8.7 |

2.6B 的 KV 只有 **16 KiB/token**（30 层里只有 8 层是真注意力），
13.8k 时 KV 才 0.21 GiB = 权重流量的 12%。按带宽算只该掉到 8.7，
实测 4.1 —— **差了 2 倍**。原因是长 KV 是 8 层 × 8 个 KV 头的跨步小读，
访存效率上不去，前端拿不满 ~15 GB/s。**L_half=102k 这个公式值在这里完全没用。**

所以你问的两个提速手段：

- ❌ **减小 `-c`：没用。** decode 只读实际存在的 KV，`-c` 改的是天花板不是现状。
- ✅ **砍实际对话长度：有用。** 7k→6.2、3.6k→8.7、1.9k→9.6。
  手段是 `--context-shift` + 客户端别把整份历史原样回灌 + 控制单文件读取量。
- ⚠️ KV 量化在这个模型上收益 <10%（KV 只占 12% 流量），且会伤 prefill，
  本轮没测完，优先级低。`ctxtune.sh` 是为这件事写的批量对比脚本。
  **补测结论见 4.17**：伤 prefill 的是**旧引擎**（b10743 q8_0 把 pp 从 67.2 打到
  55.2），换 ik_llama.cpp 后同一个 q8_0 是 88.5，比 f16 还快。机制不一样，
  别把这里的结论搬到新引擎上。

**投机解码：这条路是堵死的。** `LFM2.5-1.2B` 的词表 65536、
`LFM2.5-2.6B` 是 128000，而且**前 65536 个 token 一个都不重合**（不是子集关系，
`tokenizer.ggml.tokens` 逐个比对过）。拿 1.2B 当 draft 只会吐垃圾，
llama.cpp 也不做词表映射。这和第 6 节记录的「1.2B 给 8B-A1B 当 draft」是同一个死因。

**剩下的唯一真杠杆是减权重字节数**：换更小的量化（Q4_K_M 1.55 GiB → Q3_K_M 约 1.1 GiB，
理论上 +25%），但这台机器上 IQ* 系列没有 batched GEMM（见 4.4），
K-quant 里能省的只有 Q3/Q2，代价是 2.6B 仅有的那点路径准确度。

### 4.16 SakThai Context 0.5B：最快，但弱到只能只读

`Nanthasit/sakthai-context-0.5b-merged` = Qwen2.5-0.5B-Instruct + 2309 行
tool-calling SFT（`sakthai-0.5b-Q4_K_M.gguf`，379 MiB）。装完实测的结论是
**「速度白给，可靠性白送」**：3.4 倍速度换 0 个能落盘的文件。

**速度**（`f16 / t 4`，`bench_ctx2.py`；0.6k–7.5k 取自 `-c 8192` 那轮，14.8k 取自 `-c 16384` 那轮）：

| KV 实际长度 | prefill | decode | 对比 2.6B |
|---|---|---|---|
| 0.6k | 213 t/s | **35.7 t/s** | 10.1 |
| 2.0k | 167 | 23.8 | 9.6 |
| 7.5k | 132 | 17.9 | 6.2 |
| 14.8k | 90 | 9.7 | 4.1 |

`llama-bench tg256` 上限 38.3 t/s（t=4；t=6 只有 29.5）。
24 层**全部**存 KV（12 KiB/token，没有混合注意力的便宜可占），`L_half=32k`，
所以 8k 之前基本不掉速、过了 8k 掉得比 LFM2.5-1.2B 快得多。
工作集不到 1 GiB，`mlock` 无收益，KV 量化无收益。

**核心坑：它给 tool_call 套的壳每次都不一样，llama.cpp 只认其中一种。**
`sak_diag.py` 固定 prompt、每个格式跑 8 次的实测结果：

| 模型吐出的形态 | llama.cpp 原生解析 |
|---|---|
| `<tool_call>{"name":…}</tool_call>` | ✅ 唯一认的 |
| `<tools>{"type":"function","function":{…}}</tools>` | ❌ 0/8 |
| `{"name":"list_dir","arguments":{…}}`（裸 JSON） | ❌ 0/8 |
| `tools[name: 'list_dir', arguments: {…}]`（Python 字面量） | ❌ 0/8 |
| `<tool>list_dir</tool>{"arguments":{…}}`（模型卡上的训练格式） | ❌ 0/8 |
| `FUNCTION: list_dir\nPARAMETERS: {…}` | ❌ 0/8 |

而且**换模板救不回来**：另写的严格版 `sak_template.jinja`（把规则和正反示例塞进
system）实测同样 0/8，temp 降到 0.1 也一样。结论是**格式不稳是模型本身的问题**，
不是模板的问题 —— 所以别再跟 `--chat-template-file` 较劲，改成客户端归一化。

**修法：`sakfix.py` 归一化代理（8085 → 8084，纯客户端，不改模型不改 server）。**
它做五件事：

1. **格式归一**：上面 5 种野格式 + `<functions>call</functions> 以 X 函数调用`
   全部抠成标准 OpenAI `tool_calls`（JSON / Python 字面量 / 单引号都收）；
2. **截废话**：按 `\nuser\n` / `\nassistant\n` / `<|im_end|>` 等轮次标记切掉自问自答，
   再切退化重复（实测 `write_file` 的 content 里会冒出 400 个连续 `0`）；
3. **合法性校验**：工具名不在本次 `tools` 里就丢（防幻觉工具名），
   `arguments` 为空也丢（多半是 schema 回显，不是真调用）；
4. **one-shot 预热**：在 history 最前面插一条「调用 → 工具结果 → 一句话收尾」示例，
   内容只由 `tools` 决定（逐字节固定，所以照样命中前缀缓存），顺手教会它「答完就停」；
5. **修复回合**：有调用意图却解析不出工具名时，用 `json_schema` 约束重问一次，
   `name` 锁在本次 tools 的 enum 里，参数让模型自己填（不替它猜，猜出来的假路径更危险）。

**顺带修掉一个通用 bug：llama.cpp 对「历史 tool_calls 参数非法」直接返 HTTP 500。**
模型写 `write_file` 的长 content 撞上 `max_tokens`，`arguments` 就被截成
`{"path":"notes/a.md","content":"第一行\n- 000…`（未闭合字符串）。下一轮请求带着
这段历史进来，llama.cpp 渲染模板前会**硬解析历史 tool_calls**，解析不了就 500 ——
而且发生在 prefill **之前**，所以重试、降 `repeat_penalty`、`tool_choice=required`、
甚至去掉 `tools` 全都没用（4/4 实测全 500）。`sakfix.py` 在请求进上游之前把历史里每条
`arguments` 校验一遍：能补引号就补（保留真实内容），补不了就按 schema 降级成
`{"path": null, …}`，保证模板渲染永远不炸。**这个坑对任何模型都成立**，只要它的
工具参数可能被 `max_tokens` 截断。

配套的 flag 坑仍在：`--chat-template` 收的是**内联字符串**，文件要用
`--chat-template-file`；传错模型会顺着编「sak_template 2.0 发行说明」。

**能到什么程度**：

**直连 8084（原样）与走 8085（sakfix 代理）对比**：

| 测试 | 8084 原样 | 8085 经代理 |
|---|---|---|
| `sak_oneshot.py`（3 工具 + one-shot） | 3/4，工具名全对 | ✅ **4/4**，含「不该调工具」那题 |
| `agent_loop.py`（3 工具真实 agent 循环） | ❌ **0/4**，退化成 `list_dir -path=models/` 伪命令行 | ✅ **5 轮全跑完，4/4 工具调用正确**，平均每轮 1 个 call，30.1 t/s，前缀缓存 91.7% |
| `fs_probe.py` 明确轮 | ❌ FAIL，编 `/home/user/notes.txt` | 仍然会编 POSIX 路径（**没救，这是 0.5B 的上限**） |
| `sak_qa.py` 中文分析 | 能读懂代码，但**编造不存在的 API**、不遵守长度、不停 | 同左（不给工具时代理只透传） |

`sak_unit.py` 是代理归一化逻辑的单测（26 条，含上面 5 种野格式、幻觉工具名、
退化重复、历史参数截断修复），秒级、不碰模型，**改 `sakfix.py` 后先跑它**。

**它不会自己停。** `finish_reason` 常年 `length`，会自问自答继续编
`user\n…\nassistant`（temp 0.3 时一口气吐过 19 个 tool_call）。
客户端**必须**带 `stop=["<|im_end|>","\nuser","\nsystem"]` 且 `max_tokens ≤ 200~300`。
采样用 **temp 0.1**（0.7 会连吐 2-3 个重复 tool_call）。

另一条能用的路：`tool_choice` 传具体函数对象时，llama.cpp 会启用 grammar
约束，输出**保证**是合法 `tool_call`（实测 `list_dir({"path":"models"})` 正确），
但它约束完还会接着絮叨，仍需 stop 兜底。

**结论：`tool_calls` 恒为 null 的问题已经解决**（走 8085 代理，4/4 + 5/5 轮），
**路径幻觉没解决** —— 只读查询 / 分类 / 摘要可以用，真要落盘写代码仍然回
`agent-lfm26.cmd`。如果只是想要「更快的小模型」，`agent-lfm12-fast.cmd`
（1.2B，21.7 t/s，2k→16k 零衰减，路径偶有笔误）严格优于它。

### 4.17 自编译 ik_llama.cpp：2.6B 的 FA + 对称 KV 量化

2.6B 的旧引擎预填充只有 65.5 t/s（4.15 的表格），而 agent **每一轮都要等冷预填充**，
所以预填充比 decode 更值钱。换引擎前先说清两件事：

1. **FA 和量化 KV 在 ik_llama.cpp 里是绑定的。** 这份 fork 的 CPU FlashAttention
   kernel（F16/Q4_0/Q8_0/Q6_0）是量化 KV 的唯一入口：`-fa 0 -ctk q8_0` 直接
   `failed to create context`。所以「开 FA」不是可选项，是用 q8_0 的前提。
2. **「对称」指 q8_0**，不是 q4_1/q5_1。每个 32 元素块一个 fp16 scale，
   有 IQK FA kernel 实现，质量也比 q4_0 好。

实测（`llama-bench -t 4 -p 2048 -n 128 -r 2`，同一台机器同一个 GGUF）：

| 引擎 / KV | pp2048 | tg128 |
|---|---|---|
| b10743 · f16 | 67.2 | 8.76 |
| b10743 · q8_0 | 55.2 | 8.68 |
| ik · f16 | 84.8 | 8.61 |
| **ik · q8_0（agent-lfm26.cmd 现在用这个）** | **88.5** | 8.61 |
| ik · q4_0 | 83.6 | 8.53 |
| ik · q8_0 + `-rtr` | 64.5 | 7.59（运行时 repack，纯 CPU 反而亏，别开） |

三点结论：

- **旧引擎上量化 KV 是净亏**（55.2 vs 67.2，因为它在 O(n²) 路径里先反量化），
  这就是 4.15 里「KV 量化优先级低」的原因；**ik 上变成净赚**（88.5 > 84.8），
  因为 FA kernel 直接按量化块读 KV。所以旧注释「f16 赢」在新引擎上反了。
- **decode 没变**（8.6 vs 8.8），依旧在带宽墙上 4.15 说过的天花板里。
  换引擎换来的是预填充 +32%（对 f16）、对同配置旧引擎 +60%（q8_0），外加 KV 内存减半。
- 线程：`-pg 0,128` 空 KV 下 t=4 8.31、t=5 8.96、t=6 8.92，脚本仍保守用 t=4
  （跟旧引擎实测一致：t=6 时内存总线抢带宽反而掉）。

端到端（`agent_loop.py 5`，端口 8080）：冷预填充 869 token **10.12 s**（旧引擎同场景
82.6 s），热轮 0.7–1.5 s，前缀复用 **85.7%**，工具 4/4 名称全对，
`tool_test.py` 3/3，`fs_probe.py` 明确轮真落盘 PASS。

**构建（这台机器上从零搞出来的，脚本在 `C:\msys64\tmp\build_ik.sh`）**：

```bash
export PATH="/mingw64/bin:$PATH"
cmake -S /d/AI/LLMlocal/ik_llama-src -B /d/AI/LLMlocal/ik_llama-src/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00" \
  -DCMAKE_CXX_FLAGS="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00" \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
  -DGGML_IQK_FLASH_ATTENTION=ON -DGGML_IQK_FA_ALL_QUANTS=ON \
  -DGGML_OPENMP=ON -DGGML_CCACHE=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON
cmake --build .../build --target llama-server llama-bench -j 6
```

- `-DGGML_NATIVE=OFF` + 显式 AVX2/FMA/F16C：要的是**可复现的 AVX2 基线**，不是
  「在这台机器上碰巧解析成什么」。
- `-DLLAMA_BUILD_EXAMPLES=ON` 是必须的：`server` 和 `llama-bench` 都挂在
  `examples/` 子目录下，关掉 examples 就 `unknown target 'llama-server'`。
- 产物 `build/bin/llama-server.exe`（45.6 MB）+ `llama-bench.exe`（37.3 MB），
  旁边要放 MSYS2 的 `libgomp-1.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll` /
  `libstdc++-6.dll`，否则启动就是 `0xC0000135`（一个字节都不输出，只返回码）。

**CLI 兼容性：换引擎必须同步改脚本。** 这个 fork 删/改了三个 flag，
照抄旧脚本会直接启动失败：

| 旧脚本（b10743） | ik_llama.cpp | 处理 |
|---|---|---|
| `--checkpoint-min-step N` | `--ctx-checkpoints-interval N` | 改名 |
| `--cache-reuse N` | **不存在** | 删 |
| `--cache-idle-slots` | **不存在** | 删 |
| `--context-shift`（无参） | `--context-shift on`（要参数） | 加值 |
| `-fa on` | `-fa 1`（`on` 也收，但 bench 用 1） | 不变也行 |

## 5. 工具脚本

| 脚本 | 用途 |
|---|---|
| `agent_loop.py [轮] [端口] [alias] [temp]` | 5 轮真实 agent 循环：工具命中率 / 收尾率 / 缓存复用 / 速度。加 `--warm` 只预填充前缀就退出 |
| `tool_test.py [端口] [alias] [temp]` | 3 用例冒烟（单工具 / 多工具选择 / 不该调），退出码 0/1 |
| `tool_stress.py [端口] [alias] [temp] [N]` | N 工具（默认 12）压力测试：同域干扰项 + enum/嵌套参数 + 3 个反诱导用例，找「工具一多就选错」的规模拐点 |
| `fs_probe.py [端口] [alias] [temp] [--clean]` | **写文件能力探针**：照抄 pi 的工具 schema，在沙箱里真的落盘。分「弱提示」（只记录它干什么）和「明确提示」（必须 write + path 带文件名 + 真落盘），退出码 0/1。`tool_test.py` 3/3 通过却写不了文件，就是靠它抓出来的 |
| `bench_decode.py [端口] [alias] [长度列表]` | decode 速度 vs 上下文长度曲线（两次请求相减）。**差分法会算出负值，别信** |
| `bench_ctx2.py [端口] [alias] [长度列表]` | 同上，但直接读 server 端 `timings`（`prompt_n` / `cache_n` / `predicted_per_second`）。**当前推荐**，还能顺带看缓存命中 |
| `ctxtune.sh "<长度列表>" "<变体1>" …` | 长上下文下批量对比 KV 量化 / 线程 / fa / mlock，每变体起一次服务跑 `bench_ctx2.py` |
| `kvcalc.py <model.gguf>` | 读 GGUF 元数据，算 KV bytes/token 和各上下文下的占用 |
| `dl.sh <owner/repo> <file.gguf> <dest> [分片数]` | 魔搭 → hf-mirror → HF 直连，8 路并行 Range，严格校验字节数 |
| `sakfix.py [上游端口=8084] [监听端口=8085] [--no-oneshot] [--no-repair]` | **SakThai 归一化代理**：客户端把 0.5B 的 5 种野 tool_call 格式统一成 OpenAI 形状，截自问自答和退化重复，修历史里被 `max_tokens` 截断的 `arguments`（否则 llama.cpp 返 500）。客户端一律指 8085 |
| `sak_unit.py` | `sakfix.py` 归一化逻辑单测（26 条，不碰模型，改完代理先跑） |
| `sak_diag.py [端口] [alias] [temp]` | 把「循环 / 返 JSON / EOS」三件事量化，打印模型原始输出 |
| `sak_qa.py [端口] [alias] [temp]` | 中文问答 / 代码分析质量（不给工具，自带 stop 序列） |
| `sak_eval.py [端口] [alias] [temp] [--stop]` | 工具调用 + 中文分析 4 用例，`--stop` 对照「它会不会自己停」 |
| `sak_oneshot.py [端口] [alias] [temp]` | 在 history 最前面插一条 one-shot 示例，测格式合规率能不能救回来 |
| `sak_forced.py [端口] [alias]` | `tool_choice` 指定函数时 llama.cpp 的 grammar 约束路径 |
| `raw_probe.py [端口] [alias] [temp] [top_k]` | 看模型吐的**原始文本**，绕过 tool_call 解析器，定位「是模型没按格式还是解析器不认」 |
| `sakstart.sh [端口] "<额外参数>"` | 一次性起 SakThai 测试服务（生产用 `agent-sakthai.cmd`） |

```bash
# 下载（默认魔搭优先 —— hf-mirror 的分片内容曾损坏过）
bash dl.sh LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf models
bash dl.sh unsloth/LFM2.5-8B-A1B-GGUF LFM2.5-8B-A1B-UD-Q4_K_M.gguf models 8

# 测一批上下文长度
bash ctxsweep.sh models/LFM2.5-8B-A1B-UD-Q4_K_M.gguf probe "-t 4 -tb 6" 2048,8192,32768
bash tbench.sh t6 "-t 6"
```

### 5.1 模型下载地址（全部实测存在，2026-10-12 逐个校验）

```bash
bash dl.sh <owner/repo> <file.gguf> models [分片数]
```

| 用途 | repo | 文件 | 大小 |
|---|---|---|---|
| **最快的只读/问答底座**（需挂 `sak_template.jinja`，见 4.16） | `Nanthasit/sakthai-context-0.5b-merged` | `sakthai-0.5b-Q4_K_M.gguf` | 379 MiB |
| **agent 首选**（8B-A1B） | `unsloth/LFM2.5-8B-A1B-GGUF` | `LFM2.5-8B-A1B-UD-Q4_K_M.gguf` | 4.95 GiB |
| agent 首选·省内存版 | `unsloth/LFM2.5-8B-A1B-GGUF` | `LFM2.5-8B-A1B-UD-IQ2_M.gguf` | 2.56 GiB |
| **最快的 agent 底座** | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | 695 MiB |
| **路径最准**（做文件操作用） | `LiquidAI/LFM2.5-2.6B-GGUF` | `LFM2.5-2.6B-Q4_K_M.gguf` | 1.56 GiB |
| 纯聊天最快 | `prism-ml/Ternary-Bonsai-1.7B-gguf` | `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 442 MiB |
| 写代码最强 | `ornith-ai/Ornith-1.5-9B-GGUF` | `Ornith-1.5-9B-Q4_K_M.gguf` | 5.50 GiB |
| 起草 / 摘要（**不会调工具**） | `openbmb/MiniCPM5-1B-GGUF` | `MiniCPM5-1B-Q4_K_M.gguf` | 656 MiB |
| 纯文本（工具调用废了） | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4.36 GiB |
| 反面教材（KV 太贵） | `unsloth/Qwen3-1.7B-GGUF` | `Qwen3-1.7B-Q4_K_M.gguf` | 1.03 GiB |

> `Nanthasit/sakthai-context-0.5b-merged` 只有一个 Q4_K_M，没有别的量化。
> 它不在魔搭上，`dl.sh` 会自动落到 hf-mirror（实测可用，397,808,352 字节校验通过）。

> ⚠️ `Qwen/Qwen3-1.7B-GGUF` 官方只发布了 **Q8_0**，没有 Q4_K_M —— 要 Q4 得用
> unsloth 的仓库。文件名大小写必须一字不差（`Q4_K_M` 不是 `q4_k_m`）。
>
> `dl.sh` 按 **魔搭 → hf-mirror → HF 直连** 依次尝试。hf-mirror 的分片内容曾损坏过，
> 魔搭才是可靠的。不带参数运行 `dl.sh` 会打印三个源的完整 URL。

---

## 6. 踩过的坑（省下重复调试的时间）

**引擎 / 硬件**
- **必须用 `Bonsai-demo/bin/cpu/`**。`bin-vulkan/` 会误上 iGPU 并报
  `ggml_vulkan: Device memory allocation ... failed`（UHD 630 报 10298 MiB 但可用 0）。
  Vulkan 全卸载实测只有 11.7 t/s，纯 CPU 反而更快。`bin-vulkan/` 故意**不**放在
  `Bonsai-demo/bin/` 下，就是为了防止启动脚本误选。
- ~~`mlock` 比 `mmap` 慢 30%，保持默认 mmap。~~ **2026-10-12 重测，这条是错的**：
  在 `-ngl 0 -fa on -t 4` 下 `llama-bench -p 0 -n 96 -r 2` 测得 mmap 9.18 → `-lm mlock`
  9.79（+6.7%）、`-lm none` 9.69、`-lm mmap+mlock` 9.64。老数字多半来自当时的
  内存压力状态。详见 4.15。**`--mlock` 可以放心开**。
- `-t 6` 一律更慢（内存总线争抢）。`-t 4` 是甜点，`-tb 6` 单独给 prefill 用全核。
  0.5B 上同样成立：t=4 38.3 t/s，t=6 只有 29.5。

**模型 / 采样**
- LFM2.5 的 DSpark 投机解码是死路：draft acceptance = 0.00000，9.9 → 1.1 t/s。
- `--spec-draft-model` 拿 1.2B 当 draft：**LFM2.5 家族内部词表就对不齐**。
  1.2B 是 65536、2.6B 是 128000、8B-A1B 也是 128000，而且 1.2B 与 2.6B 的
  **前 65536 个 token 一个都不重合**（不是子集关系）。llama.cpp 不做词表映射，
  draft 只会吐垃圾。别依赖。
- **模型给 tool_call 套的壳可能每次都不一样。** SakThai 0.5B 实测有 6 种形态
  （`<tools>{…}</tools>` / 裸 JSON / Python 字面量 / `<tool>x</tool>` /
  `FUNCTION:/PARAMETERS:` / 原生 `<tool_call>`），llama.cpp 只认最后一种，
  其余全部 `tool_calls=null`；换 chat template 也救不回来（0/8）。修法是客户端归一化
  （`sakfix.py`），见 4.16。**排查工具调用问题的第一步永远是 `raw_probe.py` /
  `sak_diag.py` 看原始文本**，别只看 `tool_calls` 是不是空。
- **历史里带一条非法 JSON 的 tool_calls，整个请求会 500。** llama.cpp 在渲染
  模板前会硬解析历史 `tool_calls`，某个 `arguments` 被 `max_tokens` 截断成
  未闭合字符串就整轮 500（**prefill 之前**，重试 / 降参数 / 去掉 tools 全无效）。
  客户端要在发请求前把历史 `arguments` 校验一遍，见 4.16 的 `sanitize_history`。
- **`--chat-template` 收内联字符串，`--chat-template-file` 才收文件路径。**
  传错的话模型会把模板正文当 prompt，然后顺着编「某某包 2.0 发行说明」。
  两种写法都实测过，报错都看不出来。
- **模型卡的「工具调用 5/5」往往是文本级验证，不是 OpenAI `tool_calls` 级。**
  SakThai 卡上写「verified 5/5 tool selection（llama.cpp, CPU）」，但它给的
  期望输出是 `<tool>get_weather</tool>` 这种裸文本 —— 换句话说它自己也知道
  解析器认不出来。看模型卡先确认它的评测到底比的是什么。
- **小模型不会自己停。** SakThai 0.5B 的 `finish_reason` 常年 `length`，
  会自问自答继续编 `user\n…\nassistant`，temp 0.3 时一轮吐过 19 个 tool_call。
  客户端必须带 `stop=["<|im_end|>","\nuser","\nsystem"]` 并压低 `max_tokens`。

**测试方法**
- **测速前先确认只有一个 llama-server**。两个抢同一批核，测出来全是假数据（见过 1.1 t/s）。
- `llama-bench -p 0` 测出的 tg 对 LFM2.5 **不一定是**假数据：2.6B 报 9.18、
  Qwen2.5-0.5B 报 38.3，都和 server 端实测（10.1 / 35.7）对得上。
  但 1.2B 上曾报 10.39 而带 `-p 2048` 是 23.87。**结论不可跨模型外推，
  最终以 server 端 `timings` 为准**（`bench_ctx2.py`）。
- `llama-bench` 不接受 `-c` / `-tb`（`-ctk`/`-ctv` 可以），它按 `n_prompt+n_gen` 定上下文。
  所以「`-c` 对 decode 的影响」在 llama-bench 上根本测不出来，只能用 `bench_ctx2.py`。
- **`bench_decode.py` 的差分法不可信**：它拿「max_tokens=1 的冷请求」和
  「同 prompt + 160 tok」两次墙钟相减，实测跑出 **-0.6 t/s** 这种负数。
  改用 `bench_ctx2.py` 直接读 `timings.predicted_per_second`。
- **不带 tools 发一句纯文本**：乱码 = 文件坏了；纯文本正常但 `tool_calls` 为空 = 才是格式/预算问题。
  这一步能省掉大量瞎猜。

**下载**
- **`cat part.*` 用的是字典序不是数字序**，≥10 片时字节块整体错位，
  **大小校验通过但内容全是垃圾**（`dl.sh` 已改为按数字序拼接）。
- **大小对 ≠ 内容对**。分片下载必须校验 sha256。
- ModelScope 的 `HEAD`/`GET` 返回 chunked 无 content-length，**要用 `-r 0-0` 从
  `Content-Range` 取总长**。
- bash 按字节偏移增量读脚本，**执行期间编辑 `dl.sh` 会让后续字节错位**。

**Windows / cmd**
- **`.cmd` 必须是纯 ASCII**：cmd.exe 用代码页 936 解码 UTF-8，会把中文字节错配成管道符
  并**真的执行注释尾巴**（`rem` 里的 `|` 也一样）。`set` 行里禁止出现 `|`。
- `echo` 里的 `>` 必须转义成 `^>`，否则整行被重定向进文件 —— 症状是工作目录里
  莫名其妙冒出 `-c`、`18)`、`1` 这类 0 字节的垃圾文件（`>f16` 被解析成了重定向）。
  2026-10-12 写 `agent-lfm26.cmd` 时又踩了一次：`echo ... max_tokens >= 1400` 生成了
  一个**57 字节**的 `1400` 文件，里面是被吞掉 `>= 1400` 的那行文字。
- **`rem` 行里的 `>` 也不安全**（旧版本这里写着「只有 `echo` 会中招」，是错的）。
  2026-10-13 实测：`set "NPAR=1"  rem slots; >1 splits CTX` 的 `>` 在引号**外面**，
  是真重定向 —— 仓库根目录那个 0 字节的 `1` 就是它，每次启动都重建一次。
  更糟的情况：若同名目标是**目录**或被别的进程锁住，整行 `set` 会失败，
  `NPAR` 变空 → llama-server 收到裸 `--parallel`。已把两个脚本的这条注释改成
  「more than 1」，并删掉垃圾文件。注意 `set "X=...>..."` **引号内**的 `>` 是安全的，
  真正危险的是引号外的（含 `echo %VAR%` 间接带出来的）。
- `%ARCH%.` 不能写成 `%ARCH%\.`（反斜杠原样传给进程，`--override-kv` 失效）。
- `for /f` 取值要加 `tokens=*`，否则带空格的 NOTE 被截断。
- PowerShell 脚本要 `-ExecutionPolicy Bypass`。
- Python 打印中文要 `PYTHONIOENCODING=utf-8`（GBK 控制台 `UnicodeEncodeError`）。
- Git bash 里 curl 传中文会变 GBK → server 报 `ill-formed UTF-8`。
  用 Python `urllib` 发请求，或 JSON 写文件（`encoding="utf-8"`）+ `curl -d @file`。
- `find` 被全局禁用（会命中 `System32\find.exe` 并从 stdin 阻塞）→ 用 `fd`。
- `nohup ... &` 起的 llama-server 随 bash 会话退出被杀 → 用
  `Start-Process ... -NoNewWindow -PassThru` + `$p.PriorityClass='BelowNormal'`。
- `Stop-Process -Name` 常杀不掉，按 PID 强杀。
- **装 C++ 编译器（编 ik_llama.cpp，见 4.17）**：
  - `choco install mingw` 在这台机器上会卡死在两件事上：包校验和
    （`hashes do not match`，只能 `--ignore-checksums`），以及 **21 KB/s 的下载**
    （102 MB 要 80 分钟）。换成 MSYS2 + 国内镜像 5 分钟搞定：
    `C:\msys64\base.sfx.exe -y -oC:\` 解压后把
    `/etc/pacman.d/mirrorlist.{mingw,msys}` 第一行改成
    `https://mirrors.ustc.edu.cn/msys2/...`，再 `pacman -S --needed --noconfirm
    mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja
    mingw-w64-x86_64-libgomp`（**libgomp 要单独装**，不在 gcc-libs 里，
    缺了 CMake 报 `Could NOT find OpenMP`，FA 少了并行收益）。
  - `Start-Process -ArgumentList 'a && b'` 会被 PowerShell 按空格拆开，
    写成脚本文件再 `bash -l /tmp/x.sh`。
- **cpp-httplib 在 MinGW 上编译失败**：`'::CreateFile2' has not been declared`。
  MinGW 只在 `_WIN32_WINNT >= 0x0A00` 时声明 `CreateFile2`，而 fork 只把
  `GGML_WIN_VER` 用在自己身上，没传给 vendor。要在
  `-DCMAKE_C_FLAGS` / `-DCMAKE_CXX_FLAGS` 里显式加
  `-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00`。
- `-DLLAMA_BUILD_EXAMPLES=OFF` 会让 `llama-server` / `llama-bench` **没有 target**
  （它们挂在 `examples/` 下），报 `ninja: error: unknown target 'llama-server'`。
- 自编的 `llama-server.exe` 缺 MSYS2 运行时 DLL 时**不报错**，只返回
  `0xC0000135`，一个字符都不输出。症状是 `./llama-server.exe -h` 完全没反应。
  把 `libgomp-1.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll` /
  `libstdc++-6.dll` 拷到 `build/bin/` 即可（拷进去之前先想好：
  仓库不入库DLL，别的机器自己装一遍）。
- `-fa on` 在 `llama-bench` 里是**无效值**：bench 只收 `0|1`，传 `on` 会被当成
  关 FA（表格里 `fa` 列显示 0）。对照实验时这会静默反转结论。

---

## 7. 详细文档

| 文档 | 内容 |
|---|---|
| [README-lfm25.md](README-lfm25.md) | LFM2.5 家族：原生 tool_calls、KV 为什么便宜（30 层只有 8 层是真注意力）、thinking 吃 max_tokens、DSpark 失败实录 |
| [README-qwen.md](README-qwen.md) | Qwen/Ornith 量化对比、KV cache 架构分析、下载分片 BUG 完整记录、sha256 校验脚本 |
| [README-bonsai.md](README-bonsai.md) | Bonsai 1.7B：32k 原生窗口为什么最快、前缀缓存四规则、调参速查、YaRN 外推的坑 |
| [Bonsai-demo/README.md](Bonsai-demo/README.md) | 上游 Bonsai 2 27B 项目文档 |
| [ATLAS-engine/README.md](ATLAS-engine/README.md) | ATLAS TQ1.0 三值（1.58-bit）推理引擎，独立 C/C++ 项目 |

---

## 8. 附：ATLAS-engine（独立子项目）

TQ1.0 三值量化推理引擎，1.58 bit/权重，5 trits/byte。**它自己的结论是
「7B 在 DDR4 上 ~3 tok/s，交互不可用」** —— 真正的产物是格式和 packer，不是引擎。
本目录里存着 `Ternary-Bonsai-4B-ATLAS.tq1.atlas`（1.47 GiB）作为格式参考。
需要编译时（要求 llvm-mingw 在 PATH）：`ATLAS-engine\compile.bat`。

---

## 维护提示

- **仓库里不包含任何权重和二进制。** `models/`（36.4 GB GGUF）、`Bonsai-demo/`
  （引擎二进制）、`llama-cpp-src/`、`ik_llama-src/`、`ATLAS-engine/` 都在 `.gitignore` 里，
  各自是独立仓库。克隆后要自己准备：
  1. 引擎二进制放 `Bonsai-demo/bin/cpu/`（见 `Bonsai-demo` 仓库的 build 说明）
  2. 2.6B 要自己编译 ik 引擎，见 4.17（含编译器安装，见 6）
  3. 权重用 `bash dl.sh <owner/repo> <file.gguf> models` 下（地址见 5.1）
- `agent-lfm26.cmd` 指向 `ik_llama-src/build/bin/llama-server.exe`，
  **没编过就会自动回退到 b10743 并打印 `[WARN]`**，此时 q8_0 KV 不再是收益而是亏
  （55.2 vs 67.2）。看到 WARN 先把 4.17 的构建跑一遍。
- `agent-bonsai8b.cmd` 依赖的 `models/Ternary-Bonsai-8B-PQ2_0.gguf` **当前不存在**，
  该脚本暂时跑不起来。
- `models/` 下还有几个未接脚本的模型：`Qwen3-1.7B-Q4_K_M.gguf`、
  `LFM2.5-8B-A1B-UD-IQ2_M.gguf`、`Ornith-1.0/1.5-9B-IQ4_XS.gguf`、
  `qwen2.5-coder-3b-instruct-q2_k.gguf`、`Bonsai-1.7B-Q1_0.gguf`。
- `agent-sakthai.cmd` 会连着起两个进程：`llama-server`（8084）+ `sakfix.py`
  归一化代理（8085）。关窗口只收 server，代理要另外
  `Stop-Process -Name python -Force`（或按脚本提示的手动重启命令）。
  8085 已经被上一次残留的代理占着时脚本会警告，先清掉再起。
- `.cmd` 全部是纯 ASCII（代码页 936 会把中文注释里的字节错配成管道符并执行），
  改这些脚本时注释也只能写英文。
