# LFM2.5 本地推理（纯 CPU，给 agent 调 tools 用）

硬件：i5-8500（6 核）/ 19.9 GB RAM / **无 NVIDIA GPU**
llama.cpp：本地 `Bonsai-demo\bin\cpu\`（PrismML fork **b10743 / commit adfffbe41**）

结论先讲：这台机器上**默认用 `LFM2.5-1.2B-Instruct`**（24 t/s，原生 tool_calls 3/3 通过）。
要参数更准（尤其是文件路径）再换 `LFM2.5-2.6B`，代价是慢 2.3 倍。

---

## 一、启动

| 脚本 | 用途 |
|---|---|
| `agent-lfm25.cmd` | **agent 用这个**。OpenAI 兼容 server，`http://127.0.0.1:8080/v1` |

顶部 `set "PICK=fast"` 切模型：

| PICK | 模型 | 权重 | alias | 特点 |
|---|---|---|---|---|
| `fast`（默认） | `LFM2.5-1.2B-Instruct` Q4_K_M | 0.68 GiB | `lfm25-1.2b` | 24 t/s，不思考，**推荐** |
| `quality` | `LFM2.5-2.6B` Q4_K_M | 1.63 GiB | `lfm25-2.6b` | 10 t/s，thinking 模型，参数更准 |

停止：`Ctrl+C`，或 `powershell -c "Stop-Process -Name llama-server -Force"`

脚本启动前调 `llama-plan.ps1` 按实时空闲内存规划：f16→q8_0→q4_0→上下文减半，
保留 10% 内存余量、不超过 `MAXRAM=6` GiB / `MAXTHREADS=5` 核。

---

## 二、实测数据（这台机器，全部 b10743 / t=4 / -c 32768 / CTK=f16）

| | 1.2B-Instruct | 2.6B |
|---|---|---|
| 权重 | 0.68 GiB | 1.63 GiB |
| **生成速度**（短 prompt） | **24.1 t/s**（端到端 5 轮均值） | 10.3 t/s |
| 生成速度（2k 上下文） | ~20 t/s | 7.5 t/s |
| **预填充速度** | **176 t/s** | 65.5 t/s |
| 峰值内存（32k + f16） | 1.4 GiB（估算，实测 RSS 约 1.6） | 2.4 GiB |
| 5 轮 agent 循环总耗时 | **15.8 s** | 201.6 s |
| tool_calls 冒烟 | **3/3 通过** | **3/3 通过** |
| 工具名正确率（5 轮） | 2/3 | 2/3 |
| 前缀缓存复用率 | 84.8% | 83.5% |
| 思考 token | 无 | 每轮 60–460 |

### 为什么快：KV cache 极便宜

`lfm2` 架构 30 层里**只有 8 层是真注意力**（GQA 32 head / 8 KV head / head_dim 64），
另外 22 层是 3-token 窗口的 short convolution，**完全不占 KV cache**。

```
KV/token = 8 层 x 8 kv-head x 64 dim x 2 (K+V) x 2 byte = 16 KiB  (f16)
```

比同尺寸 qwen3 便宜约 **9 倍**。所以 32k 窗口 f16 只要 0.5 GiB，agent 的长上下文
在这台机器上几乎不要钱 —— 这是选它当 agent 底座的主要理由，不是它跑得快。

### 内存带宽就是天花板

这台机器 DDR4 实际带宽约 16 GB/s，decode 速度完全由「权重字节数 ÷ 带宽」决定：

| 模型 | 权重 | t/s | 权重×速度 |
|---|---|---|---|
| Ternary-Bonsai-1.7B PQ2_0 | 0.44 GB | 36.25 | 15.8 GB/s |
| LFM2.5-1.2B Q4_K_M | 0.69 GB | 23.87 | 16.5 GB/s |
| LFM2.5-2.6B Q4_K_M | 1.55 GB | 9.94 | 15.3 GB/s |

**加线程没用**（2.6B：t=3/4/5 都是 10.4 t/s，t=6 反而掉到 8.56）。
想快只能减权重：换小模型，或换更激进的量化（但 LFM2.5 官方只有 Q4_K_M/Q5_K_M/Q6_K/Q8_0，
hf-mirror 上 bartowski / unsloth / LiquidAI 三个仓库都**没有** IQ2/IQ3/Q2_K/Q3_K）。

---

## 三、工具调用：原生可用

不用任何 prompt hack。模型输出 Pythonic 风格调用，llama.cpp ≥ b10455 解析成
标准 OpenAI `tool_calls`（本机 b10743 已含该解析器）。

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "lfm25-1.2b",
  "messages": [{"role":"user","content":"上海现在天气怎么样？用摄氏度"}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "查询某个城市当前天气",
      "parameters": {"type":"object","properties":{
        "city":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},
        "required":["city"]}}}]}'
```

返回 `finish_reason: "tool_calls"`，参数是合法 JSON：

```json
{"city":"上海","unit":"c"}
```

冒烟测试实测（`python tool_test.py 8080 lfm25-1.2b 0.1`）：

```
[单工具]        get_weather {"city":"上海","unit":"c"}      → 回灌后正常收尾
[多工具-需选择]  get_weather {"city":"New York"}           → 回灌后正常收尾
[不该调工具]     直接纯文本回答，没有乱调工具                  PASS
```

### 两个模型的实际差距：参数保真度

2.6B 会老老实实写完整 Windows 路径，1.2B 偶尔打错：

| 期望 | 1.2B 实际输出 | 2.6B 实际输出 |
|---|---|---|
| `D:\AI\LLMlocal\llama-plan.ps1` | `D:/AI/LLLocal/models`（打错目录） | `D:\AI\LLMlocal\llama-plan.ps1` ✓ |
| model=2.6B | `llama-3-2.6b`（幻觉） | `2.6B` ✓ |

**对做文件操作 / 命令执行的 agent 来说，1.2B 这个错拼是会真出事的**，
路径敏感的 agent 请用 `PICK=quality`，或者在客户端对参数做校验再执行。

---

## 四、参数速查

```
-ngl 0            纯 CPU（这机器没 GPU，全卸载只会更慢）
-t 4              实测最快，t=3/5 持平、t=6 更慢，空出 2 核
-tb 4 -ub 512     实测 ub512 比 2048 快
--jinja           必开：加载 GGUF 内的 chat template（5442 字节，supports_tools=true）
-rea off          思考内容走 reasoning_content 字段，不混进 content
-c 32768          1.2B 原生 32k，2.6B 原生 128k，都在原生窗口内，不需要 RoPE 外推
--parallel 1      整个窗口 + 缓存都给这一个 agent
--cache-ram 2048  前缀缓存预算 MiB
--ctx-checkpoints 32 --checkpoint-min-step 256
--cache-reuse 64  历史被改写时用 KV shifting 对齐错位前缀
--context-shift   上下文写满时滑动而不是报错
--temp 0.1 --top-k 50 --repeat-penalty 1.1   模型卡值，别改
```

**采样参数必须照模型卡**：`temp 0.1 / top_k 50 / repeat 1.1`。
不要"优化"成 qwen 风格的 0.7/0.95 —— LFM2.5 训练在低熵区间，
温度一高工具调用的 JSON 就崩。

---

## 五、前缀缓存：怎么用才快

llama.cpp 按 **token 序列**找最长公共前缀，所以客户端姿势比参数更影响体感。
实测第 2 轮起复用率 92–98%，prefill 从 10.2 s 掉到 0.2–1.1 s。四条规则
（`llm_client.py` 已帮你兜住前 3 条）：

1. **system prompt + tools 放最前，逐字节不变** — 千万别在里面插「当前时间 /
   会话 ID / 随机数」，一次就废掉整个前缀。
2. **tools 数组的顺序与数量不变** — 工具 schema 也是 prompt 的一部分。
3. **历史只追加、不重写**；工具结果按固定顺序回灌。
4. alias / temp / top_k 保持一致（alias 变了就是另一个模型，缓存不通用）。

```python
import sys; sys.path.insert(0, r"D:\AI\LLMlocal")
from llm_client import LocalClient

c = LocalClient(system=SYSTEM, tools=TOOLS)   # 前缀冻结，之后别动
c.ask("一句话说明什么是向量数据库")
c.ask_tool("列出 models 目录下的文件", {"list_dir": my_func})
print(c.report())      # 调用数 / 复用率 / 平均 prefill / 缓存省下的秒数
```

观察点：每次响应的 `timings` 里 `cache_n` = 复用前缀 token，`prompt_n` = 新算的，
`prompt_ms` 只算新 token，满缓存时接近 0。

---

## 六、脚本

| 文件 | 作用 |
|---|---|
| `agent-lfm25.cmd` | 启动 server（agent 用） |
| `llm_client.py` | 前缀缓存友好的 OpenAI 客户端封装 |
| `agent_loop.py` | 5 轮真实 agent 循环：工具命中率 / 收尾率 / 缓存复用 / 速度 |
| `tool_test.py` | 3 用例 tool_calls 冒烟，退出码 0/1 |
| `llama-plan.ps1` | 资源规划（`-Size lfm12` / `-Size lfm25`） |
| `dl.sh` | 国内镜像多源分片下载器 |
| `cache_test.py` / `ttft_test.py` / `kvcalc.py` | 缓存 / 首字延迟 / KV 占用测算 |

```bash
# 冒烟 + 完整循环
python tool_test.py  8080 lfm25-1.2b 0.1
python agent_loop.py 5 8080 lfm25-1.2b 0.1

# 下载（魔搭 → hf-mirror → HF 直连，自动降级）
bash dl.sh LiquidAI/LFM2.5-1.2B-Instruct-GGUF LFM2.5-1.2B-Instruct-Q4_K_M.gguf models
```

---

## 七、踩过的坑

### 思考模型吃掉 max_tokens（最坑的一个）

2.6B 是 **thinking 模型**：每轮先吐 60–460 token 的 `reasoning_content`，
再给答案或 tool_call，而这些 token **和答案共用同一个 `max_tokens` 预算**。

`max_tokens=300` 时三个用例全部 `finish_reason=length` + `content=''`，
看起来像「工具调用完全不支持」，其实只是思考把预算吃光了。
**2.6B 至少给 600**（长轮次给到 1400 更稳）。1.2B-Instruct 不思考，400 足够。

`chat_template_kwargs={"enable_thinking": false}` **无效** —— 实测照样思考
（257 token），LiquidAI 也没发布 2.6B 的非思考 Instruct 版（只有 1.2B-Instruct）。

### 投机解码（DSpark）是死路

官方给了 `LFM2.5-2.6B-DSpark-Q4_K_M`（0.2 GiB）配套 2.6B，宣称 ~2.6x 加速且输出完全一致。
llama.cpp 能自动识别（`auto-detected speculative type 'draft-dspark' from the draft model metadata`），
但实测：

```
draft acceptance = 0.00000 (  0 accepted / 780 generated), mean len = 1.00
```

**接受率是 0**，纯开销：9.9 t/s → 1.1 t/s。已删掉这个模型，别再试。

### `llama-bench -p 0` 测出来的 tg 是假数据

`tg128` 在 `-p 0` 下 LFM2.5 走慢路径：1.2B 报 10.39 t/s，带 `-p 2048` 才是 23.87 t/s。
**测 decode 必须带上下文**，否则差 2 倍。

### GGUF 大小对了不等于内容对

第一份 2.6B 用 hf-mirror 12 路并行 range 下的，12 个分片字节数全对、合并后
`1674455040` 一个字节不差、`llama-gguf r` 也能解析出完整张量表 —— 但**权重内容是坏的**，
输出是纯乱码（`1>5\n\n\n\n\n5C?B`）。

定位方法很简单：**不带 tools 发一句纯文本**。乱码 = 文件坏了；纯文本正常但
`tool_calls` 为空 = 才是工具格式/预算问题。这一步能省掉大量瞎猜。

从 ModelScope 重下就正常了。所以现在 `dl.sh` 默认魔搭优先。

### `.cmd` 文件里绝不能出现 `|` 和非 ASCII

cmd.exe（codepage 936）会把 UTF-8 `.cmd` 的非 ASCII 字节错配成管道符执行，
**`rem` 注释里的 `|` 也一样会执行** —— `set "PICK=fast" rem a | b` 会被拆成管道，
真的去运行 `b`。`agent-lfm25.cmd` 因此保持纯 ASCII、全文无 `|`，
连 `set "X=!Y:|=+!"` 这种替换写法都不能用（改成直接 `for /f ... do echo` 打印）。

### 测速前先确认只有一个 llama-server

两个 llama-server 抢同一批核，测出来的全是假数据（我测到过 1.1 t/s 的 2.6B）。
按 PID kill 常常杀不掉，要用：

```bash
powershell -NoProfile -Command "Get-Process llama-server -ErrorAction SilentlyContinue|Stop-Process -Force"
ps aux | grep -c "[l]lama-server"     # 必须是 1 才能测
```

### ModelScope 探测大小要用 Range，不能用 HEAD

ModelScope 对 `HEAD`/`GET` 返回 `HTTP/1.1 200 OK` 但 **chunked、没有 content-length**。
用 `-r 0-0` 从 `Content-Range: bytes 0-0/1674455040` 的 `/总长` 取值才行。

### 下载脚本运行中不能改

bash 按字节偏移增量读脚本，后台跑着的时候编辑 `dl.sh` 会让后续字节错位，
报 `syntax error near unexpected token 'then'`（分片其实已经下完）。
**别在执行期间改正在跑的 shell 脚本。**

### 其它

- 纯 CPU 反而比 GPU 快：Vulkan 全卸载只有 11.7 t/s（AMD R7 200 无 fp16、显存 841 MiB；
  Intel 核显直接加载失败）。`bin-vulkan\` 故意不放 `Bonsai-demo\bin\` 下，避免脚本误选。
- Git bash 里给 curl 传中文会变 GBK 报 `ill-formed UTF-8`：JSON 用 Python 写文件
  （`encoding="utf-8"`）+ `curl -d @file`，或用 `python` 直接发 HTTP。
- 控制台输出中文乱码是终端 codepage 的问题，不影响数据：
  `PYTHONIOENCODING=utf-8 python tool_test.py ...`。

---

## 八、参考

- 模型卡 <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct> / <https://huggingface.co/LiquidAI/LFM2.5-2.6B>
- GGUF <https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF>
- 官方 benchmark（2.6B）BFCLv4 56.88 / ToolSandbox 77.83 / AIME25 51.87，
  并明确写了「not recommended for agentic coding and knowledge-heavy tasks」
- 1.2B-Instruct 官方定位：「recommended for agentic tasks, data extraction, and RAG」
