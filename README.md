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
│   └── knowledge_bases.yaml       #   可选知识库列表 + 周报固定写入目标
├── data/sources/                  # 示例科研语料（可直接替换成你自己的材料）
│   ├── weekly_reports/            #   周报 / 导师评论
│   ├── meeting_minutes/           #   组会纪要
│   ├── experiment_logs/           #   实验记录（JSON：config + metrics）
│   └── repo/                      #   代码仓库：评测协议、结果汇总表
├── src/steproute/
│   ├── cli.py                     # Typer CLI（run / evaluate / sync / brief / minutes / memory / weekly / serve / …）
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
│   ├── connectors/                # 外部来源 → 本地语料（落盘式同步，下游零改动）
│   │   ├── base.py                #   RemoteDoc / RemoteSource 协议
│   │   ├── feishu.py              #   飞书开放平台客户端（tenant_access_token / 知识库 / 评论）
│   │   └── sync.py                #   增量水位 + YAML 头部渲染 → data/sources/
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
│   │   └── rubric.py              #   报告质量评分（基准事实从语料推导）
│   ├── assistant/                 # 科研助手业务层（CLI 与网站共用）
│   │   ├── calendar.py            #   学期 / 周次 → 周报命名（`{学年}-{学期}-week{N}`）
│   │   ├── memory.py              #   项目长期记忆：状态 / 历史决策 / 实验经验
│   │   ├── minutes.py             #   组会纪要解析（决策 + 待办）与导入
│   │   ├── todo.py                #   待办清单（手动 + 文档待办，全部可勾选/改/删）
│   │   ├── distill.py             #   从文档提炼项目状态 / 决策 / 经验 → 长期记忆
│   │   ├── summarize.py           #   简报 / 待办的 LLM 归纳（失败回退规则 + 输入指纹缓存）
│   │   ├── brief.py               #   每日工作简报（待办 / 进展 / 飞书更新）
│   │   └── weekly.py              #   周报汇总、渲染与写入飞书知识库
│   ├── server/                    # 可视化网站（独立前后端，需 --extra server）
│   │   ├── app.py                 #   FastAPI 后端（dashboard / todos CRUD / distill / weekly…）
│   │   └── static/index.html      #   单页前端（工作台 / 组会纪要 / 周报 三页签）
│   └── webui.py                   # Gradio 可视化面板（可选）
└── tests/                         # 171 个离线测试
    └── fixtures/sample_corpus/    #   示例语料（测试与演示用，不污染 data/）
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
uv sync --extra server   # + fastapi / uvicorn（跑可视化网站）
uv sync --extra ui       # + gradio（跑 Web UI）
```

### 4.3 配置 LLM API Key（**默认不落明文**）

密钥解析优先级：**显式配置（`.env` / 环境变量）→ 操作系统凭据管理器 → 本地加密密钥库**。
最省事可以写进 `.env`：`STEPROUTE_LLM_API_KEY=<key>`（明文落盘、已 gitignore）；
更安全则用 `secrets set` 存进系统凭据管理器。

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
| `STEPROUTE_LLM_API_KEY` | 空 | LLM API Key（**可选**；填写即明文落盘，优先用 `secrets`） |
| `STEPROUTE_LLM_BASE_URL` | `https://api.deepseek.com/v1` | 网关地址（vLLM 本地端点写这里） |
| `STEPROUTE_PROXY_URL` | 空（直连） | 外网代理地址 |
| `STEPROUTE_BACKEND` | `api` | 默认后端：`api` / `sim` |
| `STEPROUTE_DATA_DIR` | `data` | 语料目录 |
| `STEPROUTE_ARTIFACTS_DIR` | `artifacts` | 产物目录 |
| `STEPROUTE_CONFIG_DIR` | `configs` | 配置目录 |
| `STEPROUTE_LOG_LEVEL` | `INFO` | 日志级别 |
| `STEPROUTE_VAULT_PASSPHRASE` | — | 加密密钥库主口令 |
| `STEPROUTE_VAULT_PATH` | `~/.steproute/secrets.enc` | 密钥库路径 |
| `STEPROUTE_FEISHU_APP_ID` | 空 | 飞书自建应用 App ID（非敏感） |
| `STEPROUTE_FEISHU_APP_SECRET` | 空 | 飞书应用 App Secret（**可选**；填写即明文落盘，优先用 `secrets`） |
| `STEPROUTE_FEISHU_BASE_URL` | `https://open.feishu.cn/open-apis` | 飞书开放平台基地址 |
| `STEPROUTE_FEISHU_SITE_URL` | `https://feishu.cn` | 拼接文档链接用的站点地址 |
| `STEPROUTE_FEISHU_WEEKLY_SPACE_ID` | 空 | 周报写入的知识库 space_id |
| `STEPROUTE_FEISHU_WEEKLY_TEMPLATE` | 空 | 周报模板文档 token（留空则按关键词自动匹配） |
| `STEPROUTE_FEISHU_WEEKLY_TEMPLATE_KEYWORD` | `模板` | 自动匹配模板时使用的标题关键词 |

---

## 5. 启动步骤与操作说明

### 5.1 推荐上手顺序（离线、零成本、秒级）

```powershell
uv sync
uv run steproute ingest --data-dir tests/fixtures/sample_corpus   # 1) 看示例语料加载结果
uv run steproute search "公平对比 抽帧数 分辨率"   # 2) 看 BM25 检索效果
uv run steproute run --backend sim          # 3) 离线跑完整 Agent（验证路由全链路）
uv run steproute ui --backend sim           # 4) 可视化面板
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
| `ingest` | 打印语料统计、文档列表与抽取到的 TODO |
| `search` | BM25 检索调试 |
| `sync` | 从外部知识库（飞书）同步文档与评论到 `data/sources/`；`--wiki-space` / `--folder` 指定入口，默认增量 |
| `brief` | 生成每日工作简报（项目进展 / 飞书更新），落盘 `data/briefs/` |
| `minutes` | 导入组会纪要：`--file` / `--text`，抽取决策与待办并写入语料与长期记忆 |
| `memory` | 项目长期记忆：`show` / `state` / `decision` / `lesson` / `distill`（从文档提炼） |
| `weekly` | 周报：`preview`（本地预览）/ `publish`（写入飞书知识库） |
| `serve` | 启动可视化网站（需 `--extra server`） |
| `secrets` | `set` / `status` / `delete`；`--name llm_api_key\|feishu_app_secret` |
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

### 5.5 从飞书同步会议纪要（可选）

把知识库 / 云空间里的组会纪要与**文档评论**同步成本地语料，之后照常 `ingest / search / run`
即可——检索、TODO 抽取与路由调度**零改动**复用（`connectors/` 只负责把远端内容
落成 `data/sources/<target>/*.md`）。

```powershell
# 1) 配置飞书自建应用凭证。最省事：App ID 与 App Secret 都写进项目根目录的 .env
#    （.env 已被 .gitignore 排除，不会入库；填写 Secret 即明文落盘，安全性低于 keyring）
#      STEPROUTE_FEISHU_APP_ID=cli_xxx
#      STEPROUTE_FEISHU_APP_SECRET=xxxxxxxx
#    更安全：Secret 不写 .env，改用 secrets 写入系统凭据管理器
uv run steproute secrets set --name feishu_app_secret

# 2) 先预览将同步哪些文档（不落盘）
uv run steproute sync --wiki-space "6946843325487906839" --into meeting_minutes --dry-run

# 3) 正式同步（增量：未变化的文档不重复拉取；评论自动合并进正文）
uv run steproute sync --wiki-space "6946843325487906839" --into meeting_minutes

# 也可同步个人云空间某个文件夹（token 取自浏览器地址栏；空串表示根目录）
uv run steproute sync --folder "fldbcO1UuPz8VwnpPx5a9abcef" --into meeting_minutes

# 4) 复用现有链路
uv run steproute ingest
uv run steproute search "公平对比 抽帧数"
```

* `--wiki-space` 为知识库 space_id（"我的文档库"可填 `my_library`），会**递归**遍历子节点；
  `--folder` 为云空间文件夹 token，会**递归**遍历子文件夹；二者指定其一（`docx` 之外的表格等自动跳过）；
* `--into` 选择 `data/sources/` 下的目标目录（`meeting_minutes` / `weekly_reports` / `repo` / `notes`）；
* 评论经 `GET /drive/v1/files/:token/comments` 拉取后合并为正文末尾的 `## 评论` 段，供 TODO 抽取一并消费；
* 增量水位写入 `data/sync_state.json`；远端已删除的文档会同步清理本地副本；
* 落盘文件带 YAML 头部（`title` / `date` / `source` / `source_id` / `source_url`），抽出的证据可回溯到飞书原文。

> 应用可见性：自建应用默认看不到任何文档，需把应用**加为目标知识库/文档的协作者**；
> 若只想读**个人**云空间，按飞书文档把应用加进一个群，再将该文件夹共享给该群。

### 5.6 科研助手：简报 / 记忆 / 纪要 / 周报

```powershell
# 1) 每日工作简报（项目进展 + 飞书更新），落盘 data/briefs/
uv run steproute brief
uv run steproute brief --date 2026-10-05 --no-save     # 指定日期、只看不落盘

# 2) 项目长期记忆（本地 data/memory/，可手工编辑、可版本管理）
uv run steproute memory state --project "视频理解" --status 进行中 --stage "对齐配置" --goal "本周跑通公平对比"
uv run steproute memory decision "统一 16 帧后重跑对比"
uv run steproute memory lesson "tiny ASR 在代理下不可用，改用本地模型"

# 2.1) 从文档自动提炼「项目状态 / 决策 / 经验」写入长期记忆
uv run steproute memory distill                      # 用 LLM 提炼全部文档（条目不带知识库归属）
uv run steproute memory distill --space 7552806097184784386   # 只提炼某个知识库，并给条目打上该库归属
uv run steproute memory distill --no-llm             # 不调模型，仅按规则抽取决策/经验
uv run steproute memory show

# 3) 组会纪要导入：抽取「决策」与「待办」→ 写入语料 + 长期记忆
uv run steproute minutes --file meeting.md
uv run steproute minutes --text "结论：本轮对比不公平；下一步：统一抽帧数后重跑"

# 4) 周报：先本地预览，再写入飞书知识库
uv run steproute weekly preview
uv run steproute weekly publish --space 7671654639892171748
```

**长期记忆的「知识库归属」**（实现见 `assistant/memory.py` / `assistant/distill.py`）：

* 每条长期记忆条目（项目状态 / 历史决策 / 实验经验）都带一个 `space_id`（**知识库归属**），
  它与 `project`（**LLM 归纳出的项目名**，如 `think-with-graph`）相互独立——过滤按前者、不按后者；
* **切知识库只显示该库 + 无归属的条目**：`space_id` 匹配当前所选知识库、或 `space_id == ""`（用户手写
  `memory decision/lesson`、组会纪要导入的条目——不属于任何知识库，各库都可见）；**不含其它知识库的条目**；
* `memory distill --space <id>` 会对该库重新提炼，并把新条目打上该库归属（`source="distill"`）；
  不带 `--space` 时提炼全部文档，条目无归属；
* **历史条目会成为「无归属」（各库都可见），重新提炼一次即可带上归属**：本能力引入前的旧条目没有
  `space_id` 字段，读出时一律视为无归属，不会报错；想让它归属到某个知识库，对该库跑一次
  `memory distill --space <id>` 即可——此时会对**同文本的无归属旧条目「就地认领」**（改写为当前库归属），
  不会新增重复条目；**认领后这些内容只在该库可见，会从其它知识库的视图里消失**（这正是所需的隔离行为）；
* 简报「项目进展」在规则回退时用**知识库中文名**（`space_label`）标注作用域，缺省时退回 `space_id`。

**简报与待办由 LLM 归纳**（实现见 `assistant/summarize.py`）：

* 每日简报**只含「项目进展」与「飞书更新」**（待办已在页面待办卡片独立展示）；归纳时**直读「本周更新的文档正文」
  + 长期记忆摘要**（不再依赖「下一步：」这类固定话术的正则候选），交给模型池 `balanced` 档整理成
  **从文档内容识别出的、按重要性排序的可执行待办（≤10 条）** 与一段连贯的进展摘要（3~5 句，禁止罗列原文）；
* **简报与归纳都按所选知识库作用域**：长期记忆只有 `space_id`（知识库归属）匹配或无归属的条目才计入，
  飞书更新只保留 `notes/<space_id>/` 下的文档，避免跨知识库串味；
* 「飞书更新」= **当前自然周**（周一 ~ 周日）内被改动过的文档，取自 `data/sync_state.json`
  的增量水位，因此**只反映最后一次同步时看到的情况**——远端有新改动时请先「拉取更新」再生成简报/周报；
  文档用标题展示（旧状态没存标题时回退读落盘文件头部），知识库归属由落盘路径推导（不是飞书 token）；
* **本周该知识库没有更新的文档时不调用模型**，直接走规则回退并在界面说明，避免一次无意义的空调用；
* 未配置 LLM API Key（或调用 / 解析失败）时**自动回退规则抽取**：候选按文档日期倒序取前若干条，
  进展退化为一句「N 条决策 / M 条经验」概览（**不再倒原文碎片**），界面会标明**这次是 LLM 还是规则、以及回退原因**；
* 归纳结果按**输入指纹**（本周文档的标题 + 日期 + 截断正文 + 记忆摘要的 sha1）缓存在 `data/brief_summary.json`，
  输入没变时复用缓存、不重复调模型（文档正文改动会改变指纹，从而触发重新归纳）。

**周报命名规则**（实现见 `assistant/calendar.py`）：

| 学期 | 月份 | 中间位 | 学年取 |
| --- | --- | --- | --- |
| 春季 | 2–8 月 | `1` | 当年 |
| 秋冬 | 9 月–次年 1 月 | `2` | 学期起始年（1 月归上一年） |

周次 = 以学期起始月 1 号**当天或之后**的第一个周一为第 1 周，按自然周递增。
例：`2026-2-week5`（2026-10-05 那一周）。若与你所在单位的教务口径差一周，改 `first_week_monday()` 一处即可。

**`weekly publish` 做了什么**：在知识库内按关键词找模板文档（默认「模板」，可用 `--template` 指定 token）→ 读取其 `##` 小节作为周报骨架 → 在知识库下新建同名 `docx` → 把 Markdown 转成文档块写入。

> 需要给应用开通 **创建及编辑新版文档**、**查看/编辑知识库** 权限，并把应用加为该知识空间的**成员**（写节点要求容器编辑权限）。

### 5.7 可视化网站（独立前后端）

```powershell
uv sync --extra server     # 安装 FastAPI / uvicorn
uv run steproute serve     # 打开 http://127.0.0.1:8000
```

三个页签：

| 页签 | 内容 |
| --- | --- |
| **工作台** | 知识库选择（下拉 + 「从该知识库拉取更新」+ 「同时提炼长期记忆」）、每日工作简报、**待办清单**、项目长期记忆 |
| **组会纪要** | 粘贴纪要 → 解析出「决策」与「待办」→ 写入语料与长期记忆 |
| **周报** | 预览本周周报，一键写入**固定知识库**（页面上不提供输入框） |

**项目长期记忆**（页签「工作台」）：

* 记忆面板**按当前所选知识库过滤**（`/api/dashboard?space=<id>` 的 `memory` 段）：只显示该库 +
  无归属（手写 / 纪要导入）的条目，切知识库即切换显示，不再全局混在一起；
* 每条条目带 `space_id`（**知识库归属**），与 `project`（LLM 归纳出的项目名）无关；面板展示的统计
  （`stats`）、摘要（`summary`）、项目状态 / 决策 / 经验都按该归属过滤。

**待办清单**（[todo.py](file:///d:/Research/面试项目/steproute/src/steproute/assistant/todo.py)）：

* 手动待办（`data/todos.json`）与文档待办（从周报/纪要用例抽取）**都能勾选完成 / 编辑 / 删除**；
* 文档待办不改源 Markdown（否则会破坏语料加载），而是把「改写 / 完成 / 隐藏」记为 `overrides`，展示时叠加；
* 手动待办在前，文档待办按文档日期倒序，**最多展示 10 条**（`MAX_ITEMS`，手动待办不受限）；
* 规则抽取只吃「列表项 / 复选框 / 显式待办前缀行」，不再从正文中段抠半句话（碎片）；
* 卡片标题旁显示本次归纳的素材与结果并标明**归纳方式**：LLM 模式显示「本周 N 篇文档 · 已归纳为 M 条待办」
  并带 `已用 LLM 归纳`（绿色小标签）；规则回退显示「本周候选 N 条 · 按规则兜底」（附 `summary.note` 回退原因）；
  下方有可折叠的「全部候选」区域：未被选中的候选会逐条列出（只读），每行一个「加入待办」按钮——没有"藏起来的一堆"；
* **待办与进展都按所选知识库隔离**：文档待办只取该知识库同步来的 `notes/<space_id>/`；
  手动待办带 `space_id` 归属，只显示当前知识库与其无归属（`space_id == ""`）的条目。

**简报 / 待办归纳**（[summarize.py](file:///d:/Research/面试项目/steproute/src/steproute/assistant/summarize.py)）：

* 点「用 LLM 归纳待办」调用 `POST /api/brief/summarize`，**直读本周更新的文档正文** + 长期记忆归纳成
  可执行待办与一段项目进展摘要，并写入 `data/brief_summary.json` 缓存；
* 未配置 LLM 时回退规则模式，界面会**标明这次到底用没用模型以及回退原因**（区分「显式关闭」与「漏传后端」），
  `/api/dashboard` **只读缓存、绝不主动调模型**。

**同步时自动提炼长期记忆**（[distill.py](file:///d:/Research/面试项目/steproute/src/steproute/assistant/distill.py)）：

* 勾选「同时提炼长期记忆」后点拉取，会调用模型池 `balanced` 档把文档归纳成
  「项目状态 / 历史决策 / 实验经验」写入长期记忆（决策与经验自动去重），并顺带归纳一次简报（待办 + 进展）；
  写入的条目会带上**当前知识库归属**（`space_id`），因此只在该库（及无归属条目）内可见；
* 未配置 LLM API Key 时**自动回退规则抽取**（只抽决策/经验，不推断项目状态），并在界面上说明；
* 也可单独点「只提炼，不拉取」或走 CLI `steproute memory distill --space <id>`。

* 知识库选项与周报目标来自 [configs/knowledge_bases.yaml](file:///d:/Research/面试项目/steproute/configs/knowledge_bases.yaml)，改配置即可增删；
* 点「从该知识库拉取更新」会把该知识库文档增量同步到 `data/sources/notes/<space_id>/`，随后简报 / 待办 / 长期记忆一起刷新；
* 后端接口：`/api/dashboard`（`?space=<id>` 过滤长期记忆 / 简报 / 待办）、`/api/knowledge-bases`、`/api/brief/refresh`、`/api/brief/summarize`、`/api/todos`（GET/POST）、`/api/todos/manual/{id}`（PUT/DELETE）、`/api/todos/corpus/{key}`（PUT/DELETE）、`/api/memory`（`?space=<id>`，默认不过滤）、`/api/memory/distill`、`/api/minutes`、`/api/weekly`、`/api/weekly/publish`。

---

## 6. 接入 vLLM：只改配置，不改代码

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

## 7. 密钥安全机制

| 机制 | 实现 |
| --- | --- |
| 默认不明文落盘 | `.env` 默认只放非敏感配置，密钥走 keyring / 加密密钥库；如需本地便利，可显式把 LLM Key 或飞书 App Secret 写进 `.env`（明文、已 gitignore），但安全性更低 |
| 不硬编码 | 源码、测试、文档统一使用占位符（如 `sk-test-1234567890abcdefghij`） |
| 读取即脱敏 | `redact()` 擦除所有已知密钥及 `sk-xxx` 形态；日志过滤器全链路生效 |
| 不解析明文密钥文件（可选例外） | 除显式允许的项目根目录 `.env`（已 gitignore）外，程序不会去读任何以明文保存密钥的说明文档 |
| 安全写入 | `keyring`（Windows 凭据管理器 DPAPI）/ `vault`（PBKDF2 480k 迭代 + Fernet） |

相关测试：`tests/test_secrets.py`（含"落盘不含明文""repr/status 不泄漏"等断言）。

---

## 8. 测试

```powershell
uv run pytest -q            # 171 个测试，全部离线（不联网、不需要 API Key）
uvx ruff check src tests    # 静态检查
```

覆盖：密钥安全、代理选项与连接诊断、语料加载与 BM25、待办抽取收紧（只认列表项 / 复选框 / 显式前缀，正文碎片不再误抽）、
特征抽取、Jev 判断与升级、调度器候选/缓存/队列、8 个工具（含公平性检测）、运行时端到端与 4 种策略、
动作 JSON 解析边界、rubric 评分、飞书同步（列表/递归/增量/评论合并/鉴权）、
周次命名、长期记忆（含「知识库归属」`space_id` 过滤 / 旧格式无 `space_id` 兼容）、待办增改删（含文档待办 override / 限流 / 按知识库过滤）、纪要解析、每日简报、
简报 / 待办归纳（直读本周文档正文 + 假后端注入 + 规则回退 + 无更新文档跳过模型 + 输入指纹缓存复用 / 随正文改动失效）、长期记忆提炼（规则回退 + 去重 + 写入打上知识库归属）、
周报渲染与发布（含模板套用）、Markdown→文档块、网站 API（含按 `space` 过滤长期记忆、`/api/brief/summarize` 与候选计数）、CLI 冒烟。

---

## 9. 常见问题（FAQ）

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
并体现在 `run` / `evaluate` 的成本里；想省掉这部分开销可将 `configs/jev.yaml` 的
`mode` 改成 `heuristic`。
</details>

<details>
<summary><b>Q8. `sync` 提示"未配置飞书应用 App ID / 未找到 App Secret"或鉴权失败？</b></summary>

```powershell
$env:STEPROUTE_FEISHU_APP_ID = "cli_xxx"                    # 或运行时加 --app-id
uv run steproute secrets set --name feishu_app_secret       # 推荐：存入系统凭据管理器
uv run steproute secrets status --name feishu_app_secret    # 确认解析来源（只显示掩码）
```

鉴权 / 权限失败（HTTP 401/403 或 code 99991672）还需确认：应用已发布版本、
已申请「查看新版文档」「获取云文档中的评论」等 scope，并**已被添加为目标知识库/文档的协作者**。
先加 `--dry-run` 只列目录、不落盘。
</details>

---

## 10. 已知限制

* 真实链路需要 `--backend api`，其结果受模型随机性与 API 稳定性影响；
  想离线零成本验证全链路可用 `--backend sim`。
* 样本语料为合成的小型科研项目（13 篇文档），已从 `data/` 移到 `tests/fixtures/sample_corpus/`，
  避免污染你的运行数据；`data/sources/` 只放你自己同步/导入的真实材料。
* Jev 目前是"规则 + 特征加权 + LLM 兜底"的混合策略，尚未用真实 trajectory 数据做离线拟合；
  `configs/jev.yaml` 的权重已预留为可学习参数位。
* 单条 trajectory 内通过 `max_steps` 控制预算，暂不支持跨任务并发调度（Runtime Scheduler
  已在模型池层面维护在途请求数，具备扩展基础）。
* 外部来源同步目前支持**飞书**（知识库 / 云空间文档 + 评论）；**钉钉周报（含导师评论）**尚未接入，
  但其落盘格式与目标目录已对齐，后续按同一 `connectors/` 模式扩展即可。