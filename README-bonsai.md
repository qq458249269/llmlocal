# Bonsai 1.7B 本地推理（纯 CPU，32k 原生窗口最快）

硬件：i5-8500（6 核）/ 19.9 GB RAM / 无 NVIDIA GPU
模型：`Ternary-Bonsai-1.7B-PQ2_0.gguf`（2-bit，0.43 GB），PrismML llama.cpp fork b10743

## 一、启动

| 脚本 | 用途 |
|---|---|
| `agent-bonsai17b.cmd` | **agent 用这个**：OpenAI 兼容 server，`http://127.0.0.1:8080/v1`，模型名 `bonsai-1.7b` |
| `start-bonsai17b.cmd` | 自己聊天的交互式终端 |

停止：`Ctrl+C`（或 `powershell -c "Stop-Process -Name llama-server -Force"`）

两个脚本都调用 `llama-plan.ps1` 做资源规划：按实时空闲内存自动降级
（f16→q8_0→q4_0→上下文减半），保证留 10% 内存余量、不超过 10 GB / 5 核。

## 二、实测数据（这台机器）

| 指标 | 数值 |
|---|---|
| 生成速度（短上下文，32k 窗口） | **34 t/s**（8B 是 8.2 t/s，快 4 倍） |
| 预填充速度 | 47.9 t/s（新前缀）／约 40 t/s（含缓存创建） |
| 2.3k 上下文时的生成 | 约 15 t/s（KV 读取带宽，1.7B 只有 8 个 KV head） |
| 内存 | 0.6–1.7 GiB（32k 窗口；64k 2.46 / 128k 4.42 GiB） |
| 占用 | 4/6 核，BelowNormal 优先级，整机 CPU 约 10% |
| 首字延迟 | 冷 82.6 s → 命中后 **0.67 s**（2275 token 前缀，复用 99.2%） |
| 工具调用 | 首轮 2.83 s，回灌结果 1.13 s |
| 上下文 | 32768（= 模型原生窗口，不再外推，质量最好也最快） |
| 窗口大小对速度的影响 | **没有影响**。llama-bench tg64 实测 decode 恒定：512 → 34.8 / 2k → 34.7 / 4k → 34.9 / 8k → 34.3 t/s，因为 decode 只遍历「实际存在的」KV |
| 旧记录（已作废） | 曾经记「32k 34 / 64k 21 / 128k 18 t/s」。那是把 `-c` 上限当成了对话长度，在 b10743 上**复现不出来** |

## 三、核心参数（脚本顶部可改）

```
-ngl 0            纯 CPU
-t 4 -tb 4        实测最快（t=5/6 更慢），空出 2 核
-ub 512           比 2048 快
-ctk q4_0 -ctv q4_0   比 f16 KV 预填充快 2 倍、生成快 5 倍，内存减半
-c 32768          32k 窗口（原生）。开大会明显变慢，见下面第五节
--parallel 1      整个窗口 + 缓存都给这一个 agent
--cache-ram 2048  前缀缓存预算 MiB
--ctx-checkpoints 32
--checkpoint-min-step 256   短 prompt 也能建检查点（默认 8192 对 agent 太大）
--cache-reuse 64  历史被改写时用 KV shifting 对齐错位前缀
--context-shift   上下文写满时滑动窗口而不是报错
--override-kv qwen3.context_length=int:<CTX>   仅 CTX>32768 时才需要
--rope-scaling yarn --rope-scale <CTX/32768> --yarn-orig-ctx 32768
```

## 四、前缀缓存：怎么用才快

llama.cpp 的 prompt cache 按 **token 序列** 找最长公共前缀，所以客户端姿势
比参数更影响体感。四条规则（`bonsai_client.py` 已帮你兜住前 3 条）：

1. **system prompt + tools 放最前，逐字节不变** — 千万别在里面插「当前时间 /
   会话 ID / 随机数」，一次就废掉整个前缀。
2. **tools 数组的顺序与数量不变** — 工具 schema 也是 prompt 的一部分。
3. **历史只追加、不重写**；工具结果按固定顺序回灌。
4. alias / temp / top_p 保持一致（alias 变了就是另一个模型，缓存不通用）。

### 直接用封装好的客户端

```python
import sys; sys.path.insert(0, r"D:\AI\LLMlocal")
from bonsai_client import BonsaiClient

c = BonsaiClient(system=SYSTEM, tools=TOOLS)     # 前缀冻结
c.ask("一句话说明什么是向量数据库")
c.ask_tool("列出 D:\\AI\\LLMlocal 下有哪些文件", {"list_dir": my_func})
print(c.report())      # 调用数 / 复用率 / 平均 prefill / 缓存省下的秒数
```

### 怎么观察缓存

`/slots` 端点（服务器已开）返回当前 slot 状态；每次响应的 `timings` 里：
- `cache_n` = 直接复用的前缀 token 数
- `prompt_n` = 本次真正新算的 token 数
- `prompt_ms` = **只算新 token** 的耗时（满缓存时接近 0）
- `predicted_per_second` = 生成速度

用 `cache_test.py` / `agent_loop.py` 复现上面的数字。

## 五、调参速查

| 想做的事 | 改哪里 |
|---|---|
| 换 1-bit 模型（50 t/s，质量更差） | `set MODEL=..\models\Bonsai-1.7B-Q1_0.gguf` |
| 真的要塞 10k+ token（接受慢 40%） | `set CTX=65536`（21 t/s，2.46 GiB） |
| 真的要塞 30k+ token（一次性读大文件） | `set CTX=131072`（18 t/s，4.42 GiB，YaRN 4x 外推） |
| 长文档召回更好 | `set CTK=q8_0`（64k 就要 3.9 GiB，更慢） |
| 跑多个 agent | `set NPAR=2`（窗口会平分，且靠 unified KV 共享前缀） |
| 换端口 | `set PORT=8081` |
| 更大的前缀缓存 | `set CACHEMB=4096` |

## 六、踩过的坑（避免重复调试）

- **窗口开大不只是多占内存，还会拖慢生成**（同一请求实测：32k 34 t/s、
  64k 21 t/s、128k 18 t/s，短 prompt 差距最大；上下文本身变长后差距缩小）。
  排查过不是 YaRN 的锅（32k 原生 34.2 t/s vs 32k+YaRN 31.1 t/s，噪声内），
  纯粹是 KV buffer 变大导致的带宽/寻址开销。所以默认给 32k。
- **llama-server 会把上下文截到模型训练窗口**（32768），`--rope-scale` 单独用无效，
  必须配 `--override-kv <arch>.context_length=int:<CTX>` 才能真的开到 128k。
- **YaRN 显式加 `--yarn-ext-factor 0.0 --yarn-attn-factor 1.0` 会让输出退化**
  （出现「集合的集合」这种复读），**用默认参数**。
- 纯 CPU 反而比 GPU 快：Vulkan 全卸载只有 11.7 t/s（AMD R7 200 无 fp16、
  显存 841 MiB；Intel 核显直接加载失败）。Vulkan 版在 `bin-vulkan\`，故意不放
  在 `Bonsai-demo\bin\` 下，避免启动脚本误选并默认 `-ngl 99`。
- `.cmd` 里 `%ARCH%.` 不能写成 `%ARCH%\.`（反斜杠会被原样传给进程，override 失效）。
- `--slot-save-path`（KV 落盘、跨重启复用）只在**优雅退出**（Ctrl+C / 关窗口）时
  才写盘，强制 kill 不会；且换模型时可能吃到陈旧 KV，故未启用。
- Git bash 里给 curl 传中文会变 GBK 报 `ill-formed UTF-8`：JSON 用 Python
  写文件（`encoding="utf-8"`）+ `curl -d @file`。
- `for /f` 取值要加 `tokens=*`，否则带空格的 NOTE 会被截断。
- `llama-bench` 不支持 `-c` / `-ctk`。
