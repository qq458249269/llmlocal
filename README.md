# LLMlocal — 纯 CPU 本地大模型推理 / Agent 工具调用工作台

在**一台没有 GPU 的 Windows 机器**上，把 GGUF 模型跑成 OpenAI 兼容的本地 API，
专门用来给 agent 做工具调用（tool_calls）。所有性能数字都是在这台机器上实测的，
不是抄的规格表。

**硬件基线**：i5-8500（6 核 6 线程，AVX2，无 AVX512/VNNI）· 19.86 GiB 内存 · 无独显
（仅 UHD 630 iGPU）· DDR4 实测带宽 ~16 GB/s
**推理引擎**：llama.cpp **PrismML fork b10743 / commit adfffbe41**
（必须用 `Bonsai-demo/bin/cpu/`，见「坑」一节）

---

## 1. 三十秒上手

```cmd
:: 1. 双击一个启动脚本（保持窗口不关，关窗口 = 停服务）
agent-lfm25-8b.cmd          :: 推荐：工具调用 agent 用这个，端口 8082
agent-lfm26.cmd              :: 要写文件/改代码用这个，路径最准，端口 8080
agent-lfm12-fast.cmd         :: 只要快、不写文件用这个，端口 8083

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
| 引擎 | llama.cpp **PrismML fork build 10743 / commit adfffbe41** | Clang 20.1.8，只用 `Bonsai-demo/bin/cpu/` |
| 引擎源码 | `llama-cpp-src` @ `81ef10e`（上游 ggml-org/llama.cpp） | 改 Q2_0 支持时对照用 |
| 子项目 | `Bonsai-demo` @ `9ef3205`（PrismML-Eng/Bonsai-demo） | 二进制来源 |
| 工具链 | Python 3.14.5 · Git 2.55.0 | 打印中文需要 `PYTHONIOENCODING=utf-8` |
| 空闲内存 | 实测 3.5–8.5 GiB 波动 | 所以每个启动脚本都现场跑 `llama-plan.ps1` 降级 |

**换机器时必须重测的三件事**：`llama-bench -p 2048`（**不带 `-p` 的 tg 是假数据**）、
`llama-plan.ps1 -Size lfm12 -Ctx 16384`（planner 有按模型编的峰值表）、
`fs_probe.py`（换模型或换 harness 后工具保真度可能完全变样）。
本仓库里没有一条性能数字是抄来的，全部来自上面这台机器。

---

## 2. 模型选型（实测，先看这个表）

| 启动脚本 | 模型 | 权重 | 端口 | 生成 | 预填充 | 工具调用 | 定位 |
|---|---|---|---|---|---|---|---|
| `agent-minicpm5-1b.cmd` | MiniCPM5-1B Q4_K_M | 0.64 GiB | 8081 | **26.8 t/s** | 215 t/s | ❌ 拒绝 | 最快；起草/摘要/长上下文预热 |
| `agent-lfm12-fast.cmd` | LFM2.5-1.2B-Instruct Q4_K_M | 0.68 GiB | 8083 | **21.7 t/s** | 163 t/s | ✅ 3/3 | ⭐ **最快的 agent 底座**（2k→16k 零衰减） |
| `agent-lfm25.cmd` | LFM2.5-1.2B-Instruct Q4_K_M | 0.68 GiB | 8080 | **24 t/s** | 176 t/s | ✅ 3/3 | 快的 agent 底座 |
| `agent-lfm25.cmd` | LFM2.5-2.6B Q4_K_M（`PICK=quality`） | 1.56 GiB | 8080 | 10.3 t/s | 65.5 t/s | ✅ 3/3 | 参数更准（路径不打错） |
| `agent-lfm26.cmd` | 同上，**独立脚本**（不用切 PICK） | 1.56 GiB | 8080 | 10.3 t/s | 65.5 t/s | ✅ 3/3 | ⭐ **agent 要写文件就用它** |
| `agent-lfm25-8b.cmd` | LFM2.5-8B-A1B Q4_K_M | 4.96 GiB | 8082 | 13.5 t/s | 79.7 t/s | ✅ | ⭐ **真实工具循环首选** |
| `agent-bonsai17b.cmd` | Ternary-Bonsai-1.7B-PQ2_0 | 0.43 GiB | 8080 | **34 t/s** | 47.9 t/s | ✅ | 纯聊天最快，32k 原生窗口 |
| `agent-ornith9b.cmd` | Ornith-1.5-9B Q4_K_M | 5.50 GiB | 8080 | 2.9 t/s | 25.5 t/s | ✅ | ⭐ **写代码最强**，262k 上下文 |
| `agent-qwen7b.cmd` | Qwen2.5-Coder-7B Q4_K_M | 4.36 GiB | 8080 | 3.9 t/s | 26.4 t/s | ❌ 退化成纯文本 | 快 11% 但工具调用废了 |
| `agent-bonsai8b.cmd` | Ternary-Bonsai-8B（模型未下载） | — | 8080 | 8.2 t/s | — | — | 已被 8B-A1B 取代 |

**真实 agent 前缀端到端**（`agent_loop.py --warm`，含 system + 3 工具）：

| 模型 | 耗时 | 综合速度 | 相对 |
|---|---|---|---|
| MiniCPM5-1B | 6.3 s | 136.2 t/s | 1.0×（但不会调工具） |
| LFM2.5-8B-A1B | 21.5 s | 35.4 t/s | 3.4× |
| Ornith-1.5-9B Q4_K_M | 43.7 s | 22.0 t/s | 6.9× |
| Ornith-1.5-9B IQ4_XS | 77.0 s | 12.4 t/s | 12.2× |

### 该选哪个

- **只要快 + 要调工具** → `agent-lfm12-fast.cmd`（端口 8083）。decode 21.7 t/s 且
  **2k→16k 零衰减**，16k 上下文只占 1.2 GiB，5 轮循环 16.7 s。
- **工具调用 agent，且需要 8B 级知识** → `agent-lfm25-8b.cmd`。8.47B 总参但每 token
  只激活 1.5B，速度像 1.5B，知识像 8B。代价是 decode 慢 1.6×（21.7 → 13.8 t/s）。
- **agent 要写真代码** → `agent-ornith9b.cmd`（SWE-bench Verified 70.6），
  代价是慢 7 倍。
- **纯聊天 / 最快交互** → `agent-bonsai17b.cmd`（34 t/s，但 L>4k 就腰斩）。
- **草稿、摘要** → `agent-minicpm5-1b.cmd`（26.8 t/s，但不会调工具）。

---

## 3. 目录结构

```
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
├─ bench_decode.py              ← decode 速度 vs 上下文长度曲线
├─ cache_test.py / ttft_test.py ← 前缀复用率 / 首字延迟
├─ kvcalc.py                    ← 读 GGUF 元数据算 KV cache 占用
├─ ctxsweep.sh / tbench.sh      ← decode vs 上下文 / vs 线程数 批量测试
├─ dl.sh                        ← GGUF 下载器（魔搭→hf-mirror→HF，多路分片 + 校验）
│
├─ models/                      ← 所有 GGUF（35 GB，统一平铺，不分子目录；**不入库**）
├─ Bonsai-demo/                 ← 上游 repo（PrismML-Eng/Bonsai-demo，**不入库**）
│  └─ bin/cpu/                  ← ⭐ 实际使用的 llama.cpp 二进制（b10743）
├─ bin-vulkan/                  ← Vulkan 构建（故意不放进 Bonsai-demo/bin，避免误选）
├─ llama-cpp-src/               ← llama.cpp 源码（ggml-org/llama.cpp，参考用）
├─ ATLAS-engine/                ← ATLAS TQ1.0 三值推理引擎（独立 C/C++ 项目）
└─ .gitignore                   ← 上面四个带「不入库」的都是排除项
```

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

基于上表的实测数据新建的专用脚本（端口 8083）。`llama-bench -t 4 -ub 512 -n 32`：

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
必须在客户端校验每条路径**，或改用 `agent-lfm25.cmd` 的 `PICK=quality`。

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

## 5. 工具脚本

| 脚本 | 用途 |
|---|---|
| `agent_loop.py [轮] [端口] [alias] [temp]` | 5 轮真实 agent 循环：工具命中率 / 收尾率 / 缓存复用 / 速度。加 `--warm` 只预填充前缀就退出 |
| `tool_test.py [端口] [alias] [temp]` | 3 用例冒烟（单工具 / 多工具选择 / 不该调），退出码 0/1 |
| `tool_stress.py [端口] [alias] [temp] [N]` | N 工具（默认 12）压力测试：同域干扰项 + enum/嵌套参数 + 3 个反诱导用例，找「工具一多就选错」的规模拐点 |
| `fs_probe.py [端口] [alias] [temp] [--clean]` | **写文件能力探针**：照抄 pi 的工具 schema，在沙箱里真的落盘。分「弱提示」（只记录它干什么）和「明确提示」（必须 write + path 带文件名 + 真落盘），退出码 0/1。`tool_test.py` 3/3 通过却写不了文件，就是靠它抓出来的 |
| `bench_decode.py [端口] [alias] [长度列表]` | decode 速度 vs 上下文长度曲线（两次请求相减，不依赖 server 日志） |
| `kvcalc.py <model.gguf>` | 读 GGUF 元数据，算 KV bytes/token 和各上下文下的占用 |
| `dl.sh <owner/repo> <file.gguf> <dest> [分片数]` | 魔搭 → hf-mirror → HF 直连，8 路并行 Range，严格校验字节数 |

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
| **agent 首选**（8B-A1B） | `unsloth/LFM2.5-8B-A1B-GGUF` | `LFM2.5-8B-A1B-UD-Q4_K_M.gguf` | 4.95 GiB |
| agent 首选·省内存版 | `unsloth/LFM2.5-8B-A1B-GGUF` | `LFM2.5-8B-A1B-UD-IQ2_M.gguf` | 2.56 GiB |
| **最快的 agent 底座** | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF` | `LFM2.5-1.2B-Instruct-Q4_K_M.gguf` | 695 MiB |
| **路径最准**（做文件操作用） | `LiquidAI/LFM2.5-2.6B-GGUF` | `LFM2.5-2.6B-Q4_K_M.gguf` | 1.56 GiB |
| 纯聊天最快 | `prism-ml/Ternary-Bonsai-1.7B-gguf` | `Ternary-Bonsai-1.7B-PQ2_0.gguf` | 442 MiB |
| 写代码最强 | `ornith-ai/Ornith-1.5-9B-GGUF` | `Ornith-1.5-9B-Q4_K_M.gguf` | 5.50 GiB |
| 起草 / 摘要（**不会调工具**） | `openbmb/MiniCPM5-1B-GGUF` | `MiniCPM5-1B-Q4_K_M.gguf` | 656 MiB |
| 纯文本（工具调用废了） | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` | 4.36 GiB |
| 反面教材（KV 太贵） | `unsloth/Qwen3-1.7B-GGUF` | `Qwen3-1.7B-Q4_K_M.gguf` | 1.03 GiB |

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
- `mlock` 比 `mmap` 慢 30%，保持默认 mmap。
- `-t 6` 一律更慢（内存总线争抢）。`-t 4` 是甜点，`-tb 6` 单独给 prefill 用全核。

**模型 / 采样**
- **LFM2.5-1.2B 不带 `--jinja` 就完全调不了工具**：启动 `llama-server` 必须加
  `--jinja`，它负责加载 GGUF 里的 chat template（5442 字节，`supports_tools=true`）。
  少了这个开关，模型不会报任何错，只是把调用当普通文本吐回来 —— `tool_calls` 恒为空，
  看起来像「模型不支持工具」。所有 `agent-*.cmd` 都已默认带上。
- **thinking 模型会吃光 `max_tokens`**
  与答案共用预算。`max_tokens=300` 会让三个用例全部 `finish_reason=length` + `content=''`，
  看起来像「不支持工具调用」。**至少给 600**（长轮次 1400）。
- `chat_template_kwargs={"enable_thinking": false}` 对 2.6B **无效**，实测照样思考。
- Ornith 是 reasoning 模型，一句话烧 2967 个 thinking token（≈16 分钟）。
  必须 `--reasoning-budget 256`。被砍掉的思考尾巴会泄漏进 `content`，客户端需按 `</think>` 切分。
- 采样参数必须照模型卡（LFM2.5-1.2B/2.6B：`temp 0.1 / top_k 50 / repeat 1.1`）。
  「优化」成 qwen 风格的 0.7/0.95 会让工具调用的 JSON 直接崩。
- MiniCPM5-1B 拒绝输出 tool_calls JSON（模型卡推荐 SGLang 后端 / XML 模板）。
  能聊天、能算数、能写简单代码，但**知识问答会胡说**，且不能驱动 agent。
- **Qwen2.5-Coder-7B 的工具调用会退化成裸文本**（不是模板问题、不是量化问题）。
  本 fork 下要自己写正则适配，不值得。
- LFM2.5 的 DSpark 投机解码是死路：draft acceptance = 0.00000，9.9 → 1.1 t/s。
- `--spec-draft-model` 拿 1.2B 给 8B-A1B 做 draft：词表 65536 vs 128000 对不齐，别依赖。

**测试方法**
- **测速前先确认只有一个 llama-server**。两个抢同一批核，测出来全是假数据（见过 1.1 t/s）。
- `llama-bench -p 0` 测出的 tg 是假数据（LFM2.5 走慢路径，1.2B 报 10.39，
  带 `-p 2048` 才是 23.87）。**测 decode 必须带上下文**。
- `llama-bench` 不支持 `-c` / `-ctk`。
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
  `rem` 行里的 `>` 没事，只有 `echo` 会中招。2026-10-12 写 `agent-lfm26.cmd` 时
  又踩了一次：`echo ... max_tokens >= 1400` 生成了一个**57 字节**的 `1400` 文件，
  里面是被吞掉 `>= 1400` 的那行文字。
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

- **仓库里不包含任何权重和二进制。** `models/`（36 GB GGUF）、`Bonsai-demo/`
  （引擎二进制）、`llama-cpp-src/`、`ATLAS-engine/` 都在 `.gitignore` 里，
  各自是独立仓库。克隆后要自己准备：
  1. 引擎二进制放 `Bonsai-demo/bin/cpu/`（见 `Bonsai-demo` 仓库的 build 说明）
  2. 权重用 `bash dl.sh <owner/repo> <file.gguf> models` 下（地址见 5.1）
- `agent-bonsai8b.cmd` 依赖的 `models/Ternary-Bonsai-8B-PQ2_0.gguf` **当前不存在**，
  该脚本暂时跑不起来。
- `models/` 下还有几个未接脚本的模型：`Qwen3-1.7B-Q4_K_M.gguf`、
  `LFM2.5-8B-A1B-UD-IQ2_M.gguf`、`Ornith-1.0/1.5-9B-IQ4_XS.gguf`、
  `qwen2.5-coder-3b-instruct-q2_k.gguf`、`Bonsai-1.7B-Q1_0.gguf`。
