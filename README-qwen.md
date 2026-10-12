# Qwen Coder / Ornith 量化模型 实测报告

机器：i5-8500（6 核 6 线程，无 HT）· 19.86 GiB 内存 · **无独显**（仅 UHD 630 iGPU）
实测内存带宽 ~16 GB/s → decode 上限 ≈ 权重字节 ÷ 16 GB/s
llama.cpp：**PrismML fork b10743 / adfffbe41**，必须用 `Bonsai-demo/bin/cpu/`
（`bin-vulkan/` 会误上 iGPU 并报 `ggml_vulkan: Device memory allocation ... failed`）

---

## 1. 结论：能用哪个

| 模型 | 量化 | 文件 | 权重 | tg t/s | pp t/s | 结论 |
|---|---|---|---|---|---|---|
| **Ornith-1.5-9B** | IQ4_XS | 5.24 GB | 4.83 GiB | **3.51** | 14.0 | ✅ **首选**，工具调用唯一可用的 |
| Ornith-1.0-9B | IQ4_XS | 5.20 GB | 4.83 GiB | 3.18 | 10.63 | ✅ 备选 |
| Qwen2.5-Coder-7B | Q4_K_M | 4.68 GB | 4.36 GiB | **3.90** | 26.39 | ⚠️ 快 11%，但工具调用废了 |
| Qwen2.5-Coder-3B | Q2_K | 1.38 GB | 1.29 GiB | — | — | ✅ 已校验，兜底用 |

> 上表 tg 为同配置对比值。实际运行配置见 §7（256k 上下文 + q4_0 KV，tg 3.51 t/s）。

Ornith 其余尺寸**全部放不下**：

- **31B Dense** — Q4 约 16 GB
- **35B MoE** — 最小 IQ2_M 11.66 GB（IQ1_S 7.48 GB 但质量不可用）
- **397B MoE** — 没戏

本机 19.86 GiB 里 Windows 常驻 6.7–11 GiB，实际可用只有 8–13 GiB。

---

## 2. A/B 对比（同一套 6 项测试，同一台机器）

| 测试项 | Qwen2.5-Coder-7B Q4_K_M | Ornith-1.0-9B IQ4_XS |
|---|---|---|
| 中文问答 | ✅ 6.2 s | ✅ 正常（但思考 2967 tok） |
| 英文代码（链表反转） | ✅ 21 s，代码正确 | ✅ 代码正确 |
| 工具调用 `list_dir` | ❌ **退化成纯文本** | ✅ `tool_calls` 结构化 |
| 工具调用 `get_weather` | ❌ 纯文本 | ✅ `{"city":"Shanghai","unit":"c"}` |
| 不该调工具时 | ✅ 不调 | ✅ 不调 |
| 两步 `read_file` | ❌ 纯文本 | ✅ 路径完全正确 |

**决定性差异是工具调用。**

Qwen2.5-Coder-7B 把工具调用吐成裸文本，而不是 OpenAI 的 `tool_calls` 字段：

```
--jinja (GGUF 自带模板):  ```xml\n<function name="list_dir" arguments='{"path": "D:\\AI\\LLMlocal"}'/>\n```
native 模板 (不加 --jinja):  ```xml\n<response>\n  <function_call>...\n```
```

已确认 **不是**以下原因：

- 模板问题 —— GGUF 内嵌的 chat_template 是正确的 Qwen2.5 hermes 格式
  （`<tool_call>{"name":..., "arguments":...}</tool_call>`）
- `--jinja` 开关问题 —— 两种模式都退化
- 量化问题 —— Q4_K_M 精度足够

路径内容它选得**完全正确**（`D:\AI\LLMlocal`、`D:\AI\LLMlocal\dl.sh`），纯粹是输出格式不合规。
本 fork 下需要自己写一层正则适配才能用，**不值得**。

---

## 3. Ornith-1.0-9B 详解

**架构**：`Qwen3_5ForConditionalGeneration`（Qwen3.5）
- 32 层，其中**仅 8 层 full attention**，其余 24 层是 linear attention（SSM 状态恒定，不占 KV）
- 16 attn head / 4 KV head，head_dim 256，vocab 248320
- 混合线性注意力 → **KV 仅 32 KiB/token**（Qwen2.5-Coder-7B 是 56 KiB）

**KV cache 对比（32k 上下文）**

| 模型 | KiB/token | 32k f16 |
|---|---|---|
| Ornith 9B | 32 | 1.00 GiB |
| Ornith 35B | 20 | 0.62 GiB |
| Qwen2.5-Coder-7B | 56 | 1.75 GiB |

**加载极快**：3.4 s（对比 7B 要 ~20 s）。

### 致命缺点：思考太重

Ornith 是 reasoning 模型，**一句话问答烧 2967 个 thinking token**。@3.2 t/s ≈ **16 分钟**，纯聊天不可用。

`--reasoning-budget N` 有效但非硬截断：

| 场景 | 无 budget | budget=256 |
|---|---|---|
| 中文问答 | 2967 tok / 303 s | 936 tok / 287 s |
| 英文代码 | 2458 tok / 254 s | 928 tok / 119 s |
| **工具调用** | ~150 tok | **109 tok / 50 s** |

关键：**工具调用本身只需要 ~109 thinking token**，50 s/轮，agent 场景完全可控。
被砍掉的思考尾巴会泄漏进 `content`，客户端需按 `</think>` 切分。

### 架构带来的限制

```
KV cache shifting is not supported for this context, disabling KV cache shifting
cache_reuse is not supported by this context, it will be disabled
```

- 不能用 `--cache-reuse`
- `--context-shift` 会被静默禁用，长对话需自己控制长度
- 32k 上下文实测 6 项测试全过，够用

---

## 4. GGUF 仓库

**9B 1.5（当前使用）**
- `bartowski/Ornith-1.5-9B-GGUF` — 完整低比特梯子，当前用 `IQ4_XS` 5.24 GB
  - IQ3_XS 4.56 / IQ3_M 4.72 / Q3_K_M 4.92 / **IQ4_XS 5.24** / IQ4_NL 5.48 / Q4_K_M 5.63+
- `ornith-ai/Ornith-1.5-9B-GGUF` — 官方，最低 Q4_K_M 5.78 GB

**9B 1.0**
- `KikoCis/Ornith-1.0-9B-Ollama-fixed-GGUF` — 有 Q3_K_M 起的低比特档，ModelScope 上无此 repo，走 hf-mirror
  - Q3_K_M 4.62 / **IQ4_XS 5.20** / Q4_K_M 5.63 / Q5_K_M 6.47 / Q6_K 7.36 / Q8_0 9.53 GB
- `AlexAtomic/ornith-9b-GGUF` — 自带 imatrix，另有 UD-Q4_K_XL 6.38 GB
- `ornith-ai/Ornith-1.0-9B-GGUF` — 官方，最低 Q4_K_M

**1.5 vs 1.0**

两者架构完全相同（`Qwen3_5ForConditionalGeneration`，32 层 / 8 层 full-attn / 4 KV head / head_dim 256），
所以 1.5 也能用同一套 `agent-ornith9b.cmd` 参数。1.5 是 self-improvement 的新训练版本，
知识与工具调用保真度更好。实测 1.5 IQ4_XS：tg 3.20–3.51 t/s、pp 12.8–14.0 t/s、工具调用路径零幻觉。

**35B（本机跑不动，仅记录）**
`liodon-ai/Ornith-1.0-35B-GGUF-imatrix-GGUF` — IQ1_S 7.48 / IQ2_XS 10.51 / IQ2_M 11.66 / Q2_K 12.94 / IQ3_XS 14.48 / Q3_K_M 16.76 GB

**速度估算**（16 GB/s）：9B Q3_K_M ~3.5 t/s · IQ4_XS ~3.1 · Q4_K_M ~2.8

---

## 5. `dl.sh` 分片合并 BUG（踩坑记录）

**症状**：模型能正常加载、速度正常、**文件大小校验通过**，但输出全是 `//////////` 或乱码。

**根因**：`cat "$TMP"/part.*` 用的是 bash glob 的**字典序**，不是数字序。

```
$ ls part.*
part.0  part.1  part.10  part.11  part.12  part.13  part.14  part.15  part.2 ... part.9
```

16 片时 `part.10~15` 被排到 `part.2~9` 前面 → 字节块整体错位。
**总字节数不变，所以大小校验形同虚设。**

- 8 片以内：字典序恰好 == 数字序，**不触发**（所以 3B 那次侥幸没事）
- ≥10 片：必坏

**修复**（`dl.sh` 已改）：

```bash
PARTS=()
for i in $(seq 0 $((N-1))); do [ -f "$TMP/part.$i" ] && PARTS+=("$TMP/part.$i"); done
if [ "${#PARTS[@]}" -ne "$N" ]; then
    echo "[ERR] 只找到 ${#PARTS[@]}/$N 个分片: $TMP"
    exit 1
fi
ACT=$(cat "${PARTS[@]}" 2>/dev/null | wc -c)
...
cat "${PARTS[@]}" > "$DEST/$FILE"
```

**修复后两次下载 sha256 均与 HF 完全一致**（Ornith 9B `12329ada…`、Qwen 7B `509287f7…`）。

> 教训：**大小对不等于内容对**。分片下载必须逐片或整体校验 sha256。

---

## 6. 校验脚本

```bash
# 对比 HF 上的 lfs.oid 与本地 sha256
python - <<'PY'
import json,urllib.request,ssl,hashlib
ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
REPO="KikoCis/Ornith-1.0-9B-Ollama-fixed-GGUF"; NAME="Ornith-1.0-9B-IQ4_XS.gguf"
d=json.load(urllib.request.urlopen(urllib.request.Request(
  f"https://huggingface.co/api/models/{REPO}/tree/main?recursive=1",
  headers={'User-Agent':'curl/8'}),timeout=40,context=ctx))
exp=[f['lfs']['oid'] for f in d if f['path']==NAME][0]
h=hashlib.sha256()
with open("models/"+NAME,'rb') as f:
    for b in iter(lambda: f.read(1<<22), b''): h.update(b)
print("MATCH" if h.hexdigest()==exp else "MISMATCH")
PY
```

---

## 7. 启动（推荐配置）

```cmd
agent-ornith9b.cmd
```

**统一路径**：所有模型现在都在 `D:\AI\LLMlocal\models\` 根下（不再分子目录）。

```
models/Ornith-1.5-9B-IQ4_XS.gguf        4.9G   <- 启动脚本用这个
models/Ornith-1.0-9B-IQ4_XS.gguf        4.9G
models/qwen2.5-coder-7b-instruct-q4_k_m.gguf  4.4G
models/qwen2.5-coder-3b-instruct-q2_k.gguf   1.3G
models/LFM2.5-2.6B-Q4_K_M.gguf         1.6G
models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf  698M
models/Ternary-Bonsai-4B-ATLAS.tq1.atlas  1.5G
models/Ternary-Bonsai-1.7B-PQ2_0.gguf   442M
models/Bonsai-1.7B-Q1_0.gguf           237M
```

### 实测配置（5 线程 / 8 GiB 预算）

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | `Ornith-1.5-9B-IQ4_XS.gguf` | 官方 GGUF 最低只到 Q4_K_M 5.78 GB；bartowski 仓库才有低比特梯子 |
| 上下文 | **262144** | 模型原生 `max_position_embeddings` |
| KV 量化 | **q4_0** | 256k×q4_0 = 2.0 GiB，与 128k×q8_0 同价 |
| 线程 | 5 | |
| tg | **3.51 t/s** | 比 128k+q8_0 的 3.20 **更快**（4-bit KV 少读一半带宽） |
| pp | 14.0 t/s | |
| 实测占用 | 6.87 GiB | 11.81 → 4.94 GiB 可用 |

**为什么选 256k + q4_0 而不是 128k + q8_0**：q4_0 在 256k 的 KV 占用（2.0 GiB）和 q8_0 在 128k 一样，
所以上下文翻倍**且** decode 更快。代价是 q4_0 KV 在 10 万 token 以上的 needle-in-haystack 召回会退化。
若要保底注意力保真度，把 `CTX=131072`、`CTK=q8_0` 即可（仍 6.7 GiB，tg ~3.2 t/s）。

### KV 前缀缓存

**能开的**（已开）：

```
--cache-ram 3072 --ctx-checkpoints 64 --checkpoint-min-step 512 --cache-idle-slots
```

实测 5 轮工具调用循环，命中率 86% → 92% 递增，墙钟稳定 28–32 s/轮而不随历史增长。

**开不了的**：

```
--cache-reuse  /  --context-shift
```

两者都依赖 KV shifting，而混合线性注意力上下文不支持：

```
KV cache shifting is not supported for this context, disabling KV cache shifting
cache_reuse is not supported by this context, it will be disabled
```

**后果：262144 是硬上限**——上下文满了不能滑动，必须开新会话。

其他关键参数：

```
-ngl 0                  纯 CPU
-t 5 -tb 5 -ub 512      5 线程（脚本用 -MaxThreads 5 传给 planner，否则会被降回 4）
--jinja                 必须，否则 chat template 不对
--reasoning-budget 256  必须，否则一句话烧 16 分钟
--temp 0.6 --top-p 0.95 --top-k 20
```

**一次只跑一个模型** —— 本机 19.86 GiB 装不下两个。

### 验证工具调用

用通用冒烟测试（模型无关，6 项用例：单工具 / 多工具选择 / 不该调工具 / 两跳回灌）：

```cmd
start "" /b agent-ornith9b.cmd
REM 另开一个 cmd
python tool_test.py 8080 ornith-9b 0.6
```

预期结果：Ornith 全部通过，Qwen2.5-Coder-7B 在工具调用用例上失败（退化成纯文本）。

---

## 8. 杂项坑

- `.cmd` 文件必须**纯 ASCII**（cmd.exe 用 GBK 936 代码页，会误解 UTF-8）
- `set` 行里禁止出现 `|`
- PowerShell 脚本要 `-ExecutionPolicy Bypass`
- Python 打印中文/emoji 要 `PYTHONIOENCODING=utf-8`（GBK 控制台 `UnicodeEncodeError`）
- bash/curl 传中文会触发 server `parse_error.101 ill-formed UTF-8` → 用 Python `urllib` 发请求
- `find` 被全局禁用（会命中 `C:\Windows\System32\find.exe` 并从 stdin 阻塞）→ 用 `fd`
- `nohup ... &` 起的 llama-server 随 bash 会话退出被杀 → 用
  ```
  powershell -NoProfile -Command "$a='<args>'; $p=Start-Process -FilePath '.\bin\cpu\llama-server.exe' -ArgumentList $a -NoNewWindow -PassThru; $p.PriorityClass='BelowNormal'; $p.Id | Out-File -Encoding ascii D:\AI\LLMlocal\<name>.pid"
  ```
  （`Start-Process` 在此环境会阻塞、命令常超时，但**服务其实已起来**，需另开 `curl` 确认）
- 停服务：`powershell -c "Stop-Process -Name llama-server -Force"`
- `rm -rf` 报 `Device or resource busy` → 先 `Stop-Process` 掉 curl 再删
