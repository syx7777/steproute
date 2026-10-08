# steproute

> **Step-level dynamic model routing harness for multi-step research agents**
> —— 面向科研工作流的动态模型 Harness：用 **Jev** 判断每个 Agent Step 最低需要什么能力，
> 再由 **Runtime Scheduler** 结合实时延迟、GPU 队列、前缀缓存与成本选出实际调用的模型，
> 并叠加 **Cache-aware / Failure-aware** 机制。

一句话：**不是"一个更强的模型"，而是"让每一步都用上恰到好处的模型"。**

落地场景：把散落在**周报评论 / 组会纪要 / 实验记录 / 代码仓库**里的信息串起来，
自动整理研究进展与下一步任务——一条 trajectory 十几步，难度差异极大，正是动态路由最典型的用武之地。

---

## 1. 项目背景与要解决的问题

真实的科研工作是"碎片化"的：导师的要求可能在**周报评论**里，上次组会又在**会议纪要**里提到，
最新的实验进度躺在**实验记录**中，而配置细节写在**代码仓库**里。
把这些零散信息串成"当前研究进展 + 下一步任务"，天然是一个**多步 Agent**任务。

但工程上的真正难点不是"能不能做"，而是**运行效率**：

| 问题 | 说明 |
| --- | --- |
| 步骤难度差异极大 | 从周报里提取 TODO、从实验日志读 Accuracy，小模型足够；判断"为什么没超过 baseline、下一步先验证什么"则需要强模型 |
| 统一用强模型代价高 | 一条 trajectory 十几步，全部走强模型会产生大量不必要的成本与延迟 |
| Query-level 路由不适用 | "帮我整理这周科研进展"看起来普通，执行中突然发现实验结果异常，后续就变成复杂诊断问题 |
| context length 也不是难度 | 从很长的纪要里找一个实验名很简单；判断两个接近实验为什么有差异，输入很短也很难 |
| 语义最优 ≠ 运行时最优 | 强模型可能正在排队，而当前步骤用一个稍弱的档位就能满足要求 |

因此本项目的路由粒度是 **Agent Step**，决策依据是 **Agent State**，并且把"能力判断"与"运行时调度"
拆成两层。

---

## 2. 核心设计

### 2.1 两级决策：Jev + Runtime Scheduler

```
        Agent State（步骤类型 / 深度 / 失败次数 / 无进展步数 / 子目标缺口 / 证据 / 数值冲突…）
                                    │
                          ┌─────────▼─────────┐
                          │        Jev        │   语义层：这一步"最低"需要什么能力？
                          │ heuristic / llm / │   → fast / balanced / strong
                          │      hybrid       │
                          └─────────┬─────────┘
                                    │ required_tier
                          ┌─────────▼───────────────────────────┐
                          │        Runtime Scheduler            │   系统层：运行时"最合适"的是谁？
                          │ score = w_lat·latency + w_cost·cost │   候选 = { tier ≥ required }
                          │       + w_q·queue − w_cache·hit     │   考虑 GPU 队列 / 延迟 / 前缀缓存 / 成本
                          └─────────┬───────────────────────────┘
                                    │ chosen model
                                  执行本步 → 工具调用 → 状态更新 → 下一轮
```

* **Jev** 有三种模式：`heuristic`（纯特征打分，零额外调用）、`llm`（每步问一次最便宜模型）、
  `hybrid`（默认：启发式先跑，难度分落在阈值临界区时才让 LLM 兜底）。
* **Runtime Scheduler** 只在"能力达标"的候选集合内比拼运行时指标，因此**不会**为了省钱而降级到能力不足的模型。

### 2.2 Cache-aware

科研 Agent 在同一条 trajectory 里反复使用相同的**系统 Prompt + 工具定义 + 实验背景**。
调度器按 `(前缀指纹, 模型)` 记录真实观测到的缓存命中（来自响应里的
`usage.prompt_tokens_details.cached_tokens`），命中越高的模型得分越优——
**优先继续用"已经热了"的模型，降低 TTFT**。

### 2.3 Failure-aware

若小模型**连续工具失败**或**连续多步没有推进任何子目标**，Jev 会自动把能力档上调
（`escalation.consecutive_failures` / `no_progress_streak`，默认 2），把这一步交给更强的模型重新分析。

### 2.4 关键取舍：难度由"状态"决定，而不是"长度"

`StepFeatures` 里 `context_len` 的权重被刻意压到很小（0.06），真正驱动判断的是：

| 特征 | 权重 | 含义 |
| --- | --- | --- |
| `context_len` | 0.06 | 输入长度（很弱） |
| `depth` | 0.06 | 轨迹深度 |
| `failures` | 0.08 | 历史工具失败 |
| `no_progress` | 0.08 | 无进展步数 |
| `evidence_gap` | 0.10 | 子目标有多少还没被证据覆盖 |
| `multi_source` | 0.08 | 已有证据是否来自多个来源 |
| `numeric_conflict` | 0.18 | 是否发现"配置不一致 / 数值冲突" |
| `ambiguity` | 0.14 | 结论是否歧义 |
| `staleness` | 0.08 | 上一步是否毫无产出 |

上一行 `numeric_conflict` 是很好的例子：当 `compare_experiments` 发现"方法指标落后，**但配置与 baseline 不一致**"时，
会置位该标记，Jev 立刻把后续步骤抬到强模型——因为"判断对比是否公平"本身就是高难度推理。

---

## 3. 系统架构与模块说明

```
steproute/
├── configs/                       # 全部可调参数（无需改代码）
│   ├── models.yaml                #   模型池：能力档 / 端点 / 成本 / 延迟画像  ← 换 vLLM 只改这里
│   ├── jev.yaml                   #   Jev：模式、阈值、特征权重、步骤下限、升级策略
│   ├── scheduler.yaml             #   调度器：延迟/成本/队列/缓存权重
│   ├── agent.yaml                 #   Agent：最大步数、历史窗口、输出截断
│   └── evaluation.yaml            #   仿真基准：轨迹数、步数范围、失败概率
├── data/sources/                  # 示例科研语料（可直接替换成你自己的材料）
│   ├── weekly_reports/            #   周报 / 导师评论
│   ├── meeting_minutes/           #   组会纪要
│   ├── experiment_logs/           #   实验记录（JSON：config + metrics）
│   └── repo/                      #   代码仓库：评测协议、结果汇总表
├── src/steproute/
│   ├── cli.py                     # Typer CLI（run / evaluate / benchmark / ingest / search / secrets / ui）
│   ├── config.py                  # Settings（环境变量）+ YAML 配置模型 + 能力档工具函数
│   ├── secrets.py                 # 安全密钥管理：env → keyring → 加密密钥库；读取即脱敏
│   ├── net.py                     # 代理为可选能力；连接拒绝给出可执行提示
│   ├── logging_utils.py           # 日志（统一脱敏）
│   ├── harness.py                 # 装配层：一次把全部组件拼起来
│   ├── llm/
│   │   ├── backends.py            #   APIBackend（真实 OpenAI 兼容）+ SimBackend（离线确定性）
│   │   ├── pool.py                #   模型池：静态画像 + 运行时统计（延迟 EMA / 在途 / 成本 / 缓存）
│   │   ├── schema.py              #   Usage / ChatResult / TokenLedger
│   │   └── json_utils.py          #   稳健 JSON 抽取（容忍代码围栏与前后缀说明）
│   ├── corpus/
│   │   ├── model.py               #   Document / Chunk / TodoItem / 指标方向 / 实验角色
│   │   ├── loaders.py             #   Markdown / JSON → 结构化语料（含实验 config+metrics）
│   │   └── retriever.py           #   中英文混排 BM25（中文 bigram 切词，零外部依赖）
│   ├── agent/
│   │   ├── state.py               #   AgentState / StepRecord / 证据 / 子目标 / apply_patch
│   │   ├── prompts.py             #   稳定系统前缀（利于缓存）+ 动态状态块 + 子目标模板
│   │   ├── tools.py               #   8 个工具（检索/读取/抽取/比较/登记证据/收尾）
│   │   └── runtime.py             #   **Harness 主循环**（逐步骤：状态→Jev→调度→调用→工具→更新）
│   ├── routing/
│   │   ├── features.py            #   状态 → 0~1 难度特征
│   │   ├── jev.py                 #   能力判断（heuristic / llm / hybrid + Failure-aware 升级）
│   │   ├── scheduler.py           #   运行时调度（候选打分，可解释）
│   │   └── cache.py               #   前缀缓存追踪
│   ├── evaluation/
│   │   ├── simulator.py           #   路由性能仿真器（难度-能力模型）
│   │   ├── benchmark.py           #   策略对比 + 消融 + Markdown/JSON 报告
│   │   └── rubric.py              #   报告质量评分（基准事实从语料推导）
│   └── webui.py                   # Gradio 可视化面板（可选）
└── tests/                         # 98 个离线测试
```

### 一次运行的时序

1. `HarnessRuntime.run(task)` 播种子目标（确定性模板，零模型开销）；
2. 每步：构造 `AgentState` → `extract_features` → `JevRouter.route` → `RuntimeScheduler.schedule`；
3. `APIBackend`/`SimBackend` 调用模型 → `parse_action` 解析动作 JSON；
4. `ToolExecutor.execute` 执行工具，并把副作用 `apply_patch` 回状态（含 `numeric_conflict` 等标记）；
5. 记录 `StepRecord`（模型 / 档位 / 延迟 / 成本 / 缓存 / 是否升级）→ 预测下一步类型 → 循环；
6. `finalize` 或步数耗尽 → 产出**进度报告 + 下一步任务**；`metrics` 里带完整轨迹与调度快照。

---

## 4. 环境配置与依赖安装

### 4.1 要求

| 项 | 版本 |
| --- | --- |
| Python | 3.11 ≤ version < 3.14（本项目实测 3.13.12） |
| 包管理器 | [uv](https://docs.astral.sh/uv/)（实测 0.11.28） |

### 4.2 安装

```powershell
cd steproute

uv sync                  # 核心依赖
uv sync --extra dev      # + pytest（跑测试）
uv sync --extra ui       # + gradio（跑 Web UI）
```

### 4.3 配置 LLM API Key（**不会以明文落盘**）

密钥解析优先级：**环境变量 → 操作系统凭据管理器 → 本地加密密钥库**。

```powershell
# 方式一（推荐）：写入 Windows 凭据管理器（DPAPI 加密）
uv run steproute secrets set

# 方式二：本地加密密钥库（PBKDF2-HMAC-SHA256 + Fernet），需要主口令
$env:STEPROUTE_VAULT_PASSPHRASE = "<passphrase>"
uv run steproute secrets set --backend vault

# 方式三：仅当前会话临时使用
$env:STEPROUTE_LLM_API_KEY = "<your-key>"

# 查看当前解析来源（只显示掩码，例如 sk-7***c4c6）
uv run steproute secrets status
```

> 程序**从不读取**任何"明文保存密钥"的文件，日志与异常输出统一经过 `redact()` 脱敏。

### 4.4 网络与代理（`网络和环境说明` 落地）

* **默认直连**。`api.deepseek.com` 可直连，不需要代理。
* 只有当确实需要访问外网且直连不通时，才启用本机代理（例如监听 7892）：

```powershell
# 方式一：环境变量
$env:STEPROUTE_PROXY_URL = "http://127.0.0.1:7892"

# 方式二：单次命令
uv run steproute run --proxy http://127.0.0.1:7892

# 强制直连
uv run steproute run --proxy none
```

* 端口不写死：`7892` 只是示例。代理未启动时会被识别为**确定性失败**，
  不会无意义重试，而是直接给出可执行的修复提示。

### 4.5 环境变量一览

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `STEPROUTE_LLM_API_KEY` | — | API Key（优先级最高的来源） |
| `STEPROUTE_LLM_BASE_URL` | `https://api.deepseek.com/v1` | 网关地址（vLLM 本地端点写这里） |
| `STEPROUTE_PROXY_URL` | 空（直连） | 外网代理地址 |
| `STEPROUTE_BACKEND` | `api` | 默认后端：`api` / `sim` |
| `STEPROUTE_DATA_DIR` | `data` | 语料目录 |
| `STEPROUTE_ARTIFACTS_DIR` | `artifacts` | 产物目录 |
| `STEPROUTE_CONFIG_DIR` | `configs` | 配置目录 |
| `STEPROUTE_LOG_LEVEL` | `INFO` | 日志级别 |
| `STEPROUTE_VAULT_PASSPHRASE` | — | 加密密钥库主口令 |
| `STEPROUTE_VAULT_PATH` | `~/.steproute/secrets.enc` | 密钥库路径 |

---

## 5. 启动步骤与操作说明

### 5.1 推荐上手顺序（离线、零成本、秒级）

```powershell
uv sync
uv run steproute ingest                     # 1) 看语料加载结果（13 篇文档 / 4 组实验 / 10 条待办）
uv run steproute search "公平对比 抽帧数 分辨率"   # 2) 看 BM25 检索效果
uv run steproute run --backend sim          # 3) 离线跑完整 Agent（验证路由全链路）
uv run steproute benchmark                  # 4) 策略对比 + 消融（写出 artifacts/）
uv run steproute ui --backend sim           # 5) 可视化面板
```

### 5.2 真实调用

```powershell
uv run steproute run --backend api
uv run steproute run --backend api -t "把 EXP-002 和 baseline 的对比结论重新核对一遍"
uv run steproute run --backend api --strategy always_strong   # 对照策略
uv run steproute evaluate --backend api                       # 附带 rubric 质量评分
```

### 5.3 命令清单

| 命令 | 作用 |
| --- | --- |
| `run` | 跑一次 Agent；`--strategy` 可选 `step_level`(默认) / `query_level` / `always_strong` / `always_fast`；`--out` 落盘完整 JSON |
| `evaluate` | 真实/仿真运行 + rubric 质量评分 |
| `benchmark` | 路由策略对比与三项消融，输出 `artifacts/routing_benchmark.{md,json}` |
| `ingest` | 打印语料统计、文档列表与抽取到的 TODO |
| `search` | BM25 检索调试 |
| `secrets` | `set` / `status` / `delete` |
| `ui` | 启动 Web UI（需 `--extra ui`） |

### 5.4 真实运行示例（DeepSeek `deepseek-flash` / `deepseek-v4-pro`）

```
$ uv run steproute run --backend api --max-steps 8 --quiet

    步数  8
    成本  $0.025472
档位分布  {"fast": 1, "balanced": 1, "strong": 6}
    成功  是
```

报告片段（真实输出）：

```
### 方法与 baseline 未超过的原因（关键发现）
- **已登记证据 e1**：实验对比存在配置不公平（分辨率/抽帧数不一致），
  因此当前"方法未能超过 baseline"的结论并不可靠，需统一配置后重跑公平对比。
...
## 下一步任务（优先级从高到低）
1. 统一 baseline 与方法的分辨率和抽帧数，重新跑一轮公平对比。
2. 在 32 帧 / 336 配置下补齐我们的方法。
3. 复现 baseline 时固定随机种子（seed=42）并记录评测切分。
...
```

> 注意 `fast: 1 / balanced: 1 / strong: 6`：**只有 1 步走了小模型**——
> 检索类步骤没必要动用强模型，而"比较 + 归因"这类步骤则自动落在 strong 上。

---

## 6. 结果解读：路由策略对比与消融

`uv run steproute benchmark` 会写出 `artifacts/routing_benchmark.md`。
下面是**示例运行结果**（40 条仿真轨迹，seed=20261008；可复现）：

| 策略 | 平均成本(USD) | 成本节省 | 平均延迟(s) | 延迟节省 | 平均质量 | 成功率 | 重试/条 | 缓存命中率 | 档位分布(F/B/S) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 统一强模型（对照） | 0.02576 | 0.0% | 17.06 | 0.0% | 1.000 | 100.0% | 0.00 | 88.8% | 0/0/357 |
| 统一小模型（对照） | 0.00626 | 75.7% | 1.48 | 91.3% | 0.475 | 15.0% | 4.40 | 88.7% | 353/0/0 |
| Query 级路由（对照） | 0.01030 | 60.0% | 5.33 | 68.7% | 0.757 | 47.5% | 3.08 | 89.9% | 255/79/63 |
| **步级路由（本文方案）** | **0.00873** | **66.1%** | **7.12** | **58.3%** | **0.996** | **100.0%** | **0.17** | 69.2% | 195/110/59 |
| 消融：去掉 step type | 0.01619 | 37.1% | 11.82 | 30.7% | 0.975 | 100.0% | 0.95 | 70.9% | 63/193/139 |
| 消融：去掉 Cache-aware | 0.00842 | 67.3% | 12.59 | 26.2% | 0.994 | 100.0% | 0.17 | 0.0% | 198/112/54 |
| 消融：去掉 Failure-aware | 0.00866 | 66.4% | 6.75 | 60.4% | 0.984 | 100.0% | 0.33 | 70.8% | 214/101/55 |

结论：

* **步级路由**在质量几乎不降（0.996 vs 1.000）的前提下，成本降到 **34%**、延迟降到 **42%**（相对统一强模型）。
* **统一小模型**最便宜，但质量只有 **0.475**、成功率 **15%**——"全用小模型"会把难步骤悄悄做差、并在重试里空转。
* **Query 级路由**（0.757 / 47.5%）明显弱于步级路由：最初看起来普通的任务，执行到后半段可能出现需要强模型的诊断步骤。
* **去掉 step type**（只按 context length 判难度）成本 +85%、质量 −0.02：会在"输入短但很难"的步骤上失手。
* **去掉 Cache-aware** 延迟 +77%：频繁切换模型会丢掉前缀缓存、抬高 TTFT。
* **去掉 Failure-aware** 质量 −0.012：失败后不升级，重试质量下降。

> 说明：`benchmark` 跑在 `evaluation/simulator.py` 的**显式难度-能力模型**上，
> 目的是让几十条轨迹 × 7 种策略 × 固定随机种子可复现且不烧钱；
> 真实链路由 `run --backend api` 覆盖。两者的路由/调度代码是同一套。

---

## 7. 接入 vLLM：只改配置，不改代码

底层本想用 vLLM 部署多个能力档的模型。本项目把"能力档"与"具体模型"解耦，
因此从 DeepSeek 官方切到本地 vLLM 只需要改 `configs/models.yaml`：

```yaml
models:
  - id: qwen-fast
    tier: fast
    model: Qwen2.5-1.5B-Instruct
    endpoint: http://127.0.0.1:8000/v1
    base_latency_s: 0.4
    capability: 0.45
  - id: qwen-balanced
    tier: balanced
    model: Qwen2.5-7B-Instruct
    endpoint: http://127.0.0.1:8000/v1
    capability: 0.72
  - id: qwen-strong
    tier: strong
    model: Qwen2.5-32B-Instruct
    endpoint: http://127.0.0.1:8001/v1
    max_concurrency: 2      # 供调度器估计 GPU 队列
    capability: 0.93
```

`RuntimeScheduler` 的队列惩罚、延迟估算、缓存打分全部基于这里的画像，**路由与调度逻辑无需改动**。

---

## 8. 密钥安全机制

| 机制 | 实现 |
| --- | --- |
| 不明文落盘 | `.env` 只放非敏感配置；密钥走 keyring / 加密密钥库 |
| 不硬编码 | 源码、测试、文档统一使用占位符（如 `sk-test-1234567890abcdefghij`） |
| 读取即脱敏 | `redact()` 擦除所有已知密钥及 `sk-xxx` 形态；日志过滤器全链路生效 |
| 不解析明文密钥文件 | 程序不会去读任何以明文保存密钥的说明文档 |
| 安全写入 | `keyring`（Windows 凭据管理器 DPAPI）/ `vault`（PBKDF2 480k 迭代 + Fernet） |

相关测试：`tests/test_secrets.py`（含"落盘不含明文""repr/status 不泄漏"等断言）。

---

## 9. 测试

```powershell
uv run pytest -q            # 98 个测试，全部离线（不联网、不需要 API Key）
uvx ruff check src tests    # 静态检查
```

覆盖：密钥安全、代理选项与连接诊断、语料加载与 BM25、特征抽取、Jev 判断与升级、
调度器候选/缓存/队列、8 个工具（含公平性检测）、运行时端到端与 4 种策略、
动作 JSON 解析边界、benchmark 对比与消融、rubric 评分、CLI 冒烟。

---

## 10. 常见问题（FAQ）

<details>
<summary><b>Q1. 报错"无法连接代理 … 连接被拒绝"怎么办？</b></summary>

说明 `STEPROUTE_PROXY_URL` 指向的代理没启动。本机代理不是必需项：

```powershell
uv run steproute run --proxy none        # 强制直连
# 或指定正确端口
$env:STEPROUTE_PROXY_URL = "http://127.0.0.1:<port>"
```
</details>

<details>
<summary><b>Q2. 提示"未找到 LLM API Key"？</b></summary>

```powershell
uv run steproute secrets set       # 推荐
uv run steproute secrets status    # 确认解析来源
```
</details>

<details>
<summary><b>Q3. 只想先跑通流程，不想花钱？</b></summary>

加 `--backend sim`。离线后端用确定性参考策略驱动整条链路，不发起任何网络请求，
`run / evaluate / ui` 都支持。
</details>

<details>
<summary><b>Q4. 换成我自己的科研材料？</b></summary>

把文件丢进 `data/sources/` 对应子目录即可：

* `weekly_reports/`、`meeting_minutes/`：`.md` / `.txt`，可选 YAML 头部（`title` / `date` / `author`）；
* `experiment_logs/`：`.json`，字段 `experiment_id` / `name` / `config` / `metrics` / `notes`；
* `repo/`：任意 `.md` / `.json`。

检索、TODO 抽取、公平性检测、rubric 基准事实都会自动适配新语料（无需改代码）。
</details>

<details>
<summary><b>Q5. 模型不遵守输出协议怎么办？</b></summary>

协议要求每步返回一个 JSON 动作对象。解析器 `extract_json_block` 容忍
代码围栏、前后解释文字与残缺包裹；若整步仍无法解析，会被计为**失败步**并触发
Failure-aware 升级，同时最多重试到上限。可适当提高 `configs/models.yaml` 里对应档位的
`max_tokens`（推理模型的 reasoning token 会占用额度）。
</details>

<details>
<summary><b>Q6. Web UI 启动失败？</b></summary>

`ui` 依赖可选组 `gradio`：`uv sync --extra ui`。若 7860 端口被占用：
`uv run steproute ui --port 7861`。
</details>

<details>
<summary><b>Q7. `judge` 调用会计入成本吗？</b></summary>

会计入。`Jev` 的 LLM 兜底走**最便宜的能力档**，其 token 与延迟同样写入模型池统计，
并在 `benchmark` 的成本里体现；想省掉这部分开销可将 `configs/jev.yaml` 的
`mode` 改成 `heuristic`。
</details>

---

## 11. 已知限制

* `benchmark` 的结论来自**仿真器**（显式难度-能力模型），用于可复现的策略对比；
  真实链路需要 `--backend api`，其结果受模型随机性与 API 稳定性影响。
* 样本语料为合成的小型科研项目（13 篇文档），用于演示与回归测试，
  不代表真实实验室的数据规模。
* Jev 目前是"规则 + 特征加权 + LLM 兜底"的混合策略，尚未用真实 trajectory 数据做离线拟合；
  `configs/jev.yaml` 的权重已预留为可学习参数位。
* 单条 trajectory 内通过 `max_steps` 控制预算，暂不支持跨任务并发调度（Runtime Scheduler
  已在模型池层面维护在途请求数，具备扩展基础）。