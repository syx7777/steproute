# steproute

> **用最合适的模型，跑好每一步。**
> 面向多步 Agent 的**步级动态模型路由与调度**：先判断每一步最低需要什么能力，再由 Runtime Scheduler 在够用的模型里挑出运行时最合适的那个，并叠加 Cache-aware 与 Failure-aware 机制。

一条多步 Agent 任务，往往要调用十几次甚至几十次模型，而每一步的难度差得很远：从周报里提取 TODO、从实验日志里读一个 Accuracy，小模型完全够用；但判断"这个方法为什么没超过 baseline、下一步先验证什么"，就必须上强模型。全用强模型，又贵又慢；全用弱模型，关键推理做不好。

steproute 把这件事拆成两步：**先判断"这一步至少要多强"，再在够用的模型里挑"当下最划算的那个"**。

## ✨ 为什么用 steproute

- **该省则省，该强则强**：判档粒度是"每一步"，不是"整道题"，简单步骤不浪费强模型。
- **够用才降级**：调度器只在"能力达标"的候选模型里比延迟、队列、缓存和成本，**不会为了省钱把任务交给做不动的模型**。
- **越跑越顺（Cache-aware）**：同一条轨迹里反复出现的系统提示、工具定义会被记录缓存命中，优先复用"已经热了"的模型，首字延迟更低。
- **卡住了会自救（Failure-aware）**：连续工具失败、或连续几步都没有进展时，自动升级到更强的模型重新分析。
- **难度看状态，不看长度**：输入长短几乎不影响判档（`context_len` 权重只有 0.06），真正起作用的是失败次数、无进展步数、证据缺口、数值冲突这些**当前状态**特征。
- **改配置不改代码**：模型池、路由权重、调度权重都在 `configs/*.yaml` 里；从 DeepSeek 换成本地 vLLM 只改一个 `configs/models.yaml`。
- **能离线复现**：自带确定性离线后端（`--backend sim`）和示例语料，**不联网、不花钱**就能验证整条链路。
- **还能顺手干活**：飞书文档/评论同步、每日简报、可勾选的待办、项目长期记忆、周报生成与发布，开箱即用。

## 🎯 典型适用场景

**适合你，如果你：**

- 在做**多步 Agent / 工作流**，一条任务要调用十几次以上模型，且各步难度差异明显；
- 希望在**质量与成本之间可控**，而不是"要么全强、要么全弱"；
- 打算接入**本地 vLLM 的多档模型**，想让调度跟着 GPU 队列、延迟、缓存动态选模型；
- 有**散落在多处的科研或项目材料**（周报评论、组会纪要、实验记录、代码仓库），想自动整理成进展与下一步待办。

**可能不太合适：**

- 单轮问答、单步任务——只调用一次模型，没有"逐步路由"的空间；
- 只用一个模型、也不在意成本与延迟的场景。

## 🧭 它怎么工作

```
        Agent State（步骤类型 / 轨迹深度 / 失败次数 / 无进展步数 / 子目标缺口 / 证据 / 数值冲突…）
                                    │
                          ┌─────────▼─────────┐
                          │      能力判档      │   这一步"最低"需要什么能力？
                          │ 启发式 + 可选 LLM  │   → fast / balanced / strong
                          └─────────┬─────────┘
                                    │ required_tier
                          ┌─────────▼───────────────────────────┐
                          │        Runtime Scheduler            │   运行时"最合适"的是谁？
                          │ score = w_lat·latency + w_cost·cost │   候选 = { tier ≥ required }
                          │       + w_queue·queue − w_cache·hit │   考虑 GPU 队列 / 延迟 / 前缀缓存 / 成本
                          └─────────┬───────────────────────────┘
                                    │ chosen model
                                  执行本步 → 调用工具 → 回写状态 → 下一轮
```

一次运行的顺序是：构造 `AgentState` → 抽取难度特征 → 判定这一步的最低能力档 → Scheduler 选定模型 → 调用模型 → 执行工具并把结果回写状态 → 进入下一步。

关键取舍：判档只回答"至少要多强"，Scheduler 只在**达标的候选**里做运行时优化，所以"省钱"永远不会以"能力不足"为代价。

判档本身由两部分组成，取二者中较高的档位：

1. **步骤类型的能力下限表**（硬约束）：检索、读取 → `fast`；比较、综合、收尾 → `balanced`；实验分析 → `strong`。见 [configs/jev.yaml](configs/jev.yaml) 的 `step_classifier.min_tier`。
2. **当前状态的加权分数**：把失败次数、无进展步数、证据缺口、数值冲突等特征加权，再落到 `fast / balanced / strong`。

默认是纯启发式（`heuristic`）**不调用任何模型**；`llm` 模式会每步用最便宜的模型判一次档；`hybrid` 模式只在分数落在阈值临界区时才让最便宜的模型兜底判一次。**离线后端（`--backend sim`）下不会调用任何模型。**

## 🗂 目录结构

```
steproute/
├── configs/                 # 全部可调参数（无需改代码）
│   ├── models.yaml          #   模型池：能力档 / 端点 / 成本 / 延迟画像  ← 换 vLLM 只改这里
│   ├── jev.yaml             #   判档：模式、阈值、特征权重、升级策略
│   ├── scheduler.yaml       #   调度器：延迟 / 成本 / 队列 / 缓存权重
│   ├── agent.yaml           #   Agent：最大步数、历史窗口
│   └── knowledge_bases.example.yaml   # 知识库列表模板（复制为 knowledge_bases.yaml 使用）
├── data/sources/            # 语料目录（可替换成你自己的材料）
├── src/steproute/
│   ├── cli.py               # 命令行入口
│   ├── harness.py           # 装配层：把全部组件拼起来
│   ├── llm/                 # 模型池 + 后端（真实 API / 离线 Sim）
│   ├── corpus/              # 语料加载 + 中英文 BM25 检索
│   ├── connectors/          # 飞书同步（落盘式，下游零改动）
│   ├── agent/               # 状态 / 工具 / 主循环
│   ├── routing/             # 特征 / 判档 / 调度器 / 缓存追踪
│   ├── assistant/           # 简报 / 待办 / 长期记忆 / 周报
│   └── server/              # FastAPI 后端 + 单页前端
└── examples/                # 示例语料（纯虚构数据，可直接试跑）
```

---

## 📦 部署

按下面顺序做，走完就能跑起来。**第 1～3 步不需要任何 API Key、不联网、不花钱。**

### 1. 前置条件

| 依赖 | 要求 | 说明 |
| --- | --- | --- |
| Python | 3.11 ～ 3.13 | 建议 3.12；版本超出范围 `uv` 会直接报错 |
| uv | 最新版 | 项目用它管理环境与依赖 |
| 网络 | 默认直连 | 只有访问受限时才需要代理 |
| GPU | 不需要 | 跑 Agent 只用 API；接本地 vLLM 时才需要 |

还没装 `uv` 的话，先装它：

```powershell
# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 获取代码并安装依赖

```bash
git clone https://github.com/<your-account>/steproute.git
cd steproute

uv sync                  # 核心依赖（跑 Agent / CLI）
uv sync --extra server   # 额外装 fastapi、uvicorn（用可视化网站）
uv sync --extra ui       # 额外装 gradio（用 Web UI）
```

> 只跑通流程，`uv sync` 就够了；要用网站再加 `--extra server`。

### 3. 跑通离线示例（验证部署成功 ✅）

项目自带一份示例语料（在 `examples/sample_corpus/`，纯虚构数据）。因为它不在默认的 `data/` 目录下，所以命令里要显式加 `--data-dir`：

```powershell
uv run steproute version                                        # 应输出：steproute 1.0.0
uv run steproute ingest  --data-dir examples/sample_corpus      # 看语料加载结果
uv run steproute search "baseline 对比 公平" --data-dir examples/sample_corpus   # 验证检索
uv run steproute run --backend sim --data-dir examples/sample_corpus             # 离线跑完整 Agent
```

`run` 最后会打印一段结果，看到下面这样就说明**部署成功**：

```text
───────────────────────── 运行结果 · 策略=step_level ──────────────────────────
    步数  7
    耗时  0.00 s
    成本  $0.003277
档位分布  {"fast": 3, "strong": 4}
    成功  是
```

档位分布说明它确实在按步判档：检索/读取类步骤走 `fast`，`compare` 这类要推理的步骤走 `strong`。

### 4. 配置 API Key（要做真实调用 / 简报 / 周报时才需要）

密钥默认**不落明文**，解析顺序是：`显式配置（.env / 环境变量）→ 操作系统凭据管理器 → 本地加密密钥库`。

```powershell
# 推荐：写进系统凭据管理器（Windows 走 DPAPI 加密）
uv run steproute secrets set                 # 交互式输入，不回显
uv run steproute secrets status              # 只看解析来源（显示掩码）

# 或：只对当前终端生效
$env:STEPROUTE_LLM_API_KEY = "<your-key>"

# 或：复制模板，填到 .env（.env 已在 .gitignore 里，不会入库）
Copy-Item .env.example .env
```

### 5. 换成自己的模型（可选）

默认按能力档映射到 DeepSeek 官方网关（`https://api.deepseek.com/v1`），配置在 [configs/models.yaml](configs/models.yaml)。想换成**本地 vLLM**，只改这一个文件即可，路由与调度代码不用动：

```yaml
models:
  - id: qwen-fast
    tier: fast
    model: Qwen2.5-1.5B-Instruct
    endpoint: http://127.0.0.1:8000/v1
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

### 6. 配置代理（可选，默认直连）

程序默认直连，不需要代理。确实要访问外网且直连不通时：

```powershell
$env:STEPROUTE_PROXY_URL = "http://127.0.0.1:7892"     # 端口按你的实际情况改
uv run steproute run --proxy none                      # 也可以强制直连
```

### 7. 配置自己的知识库（用网站 / 周报时才需要）

```powershell
Copy-Item configs/knowledge_bases.example.yaml configs/knowledge_bases.yaml
# 然后把自己的知识库 space_id 填进去
```

> `configs/knowledge_bases.yaml` 已被 `.gitignore` 排除，不会提交到仓库。

### ✅ 部署自检清单

- [ ] `uv run steproute version` 输出 `steproute 1.0.0`
- [ ] 离线示例的 `run --backend sim` 打印出「档位分布」
- [ ] （装了 `--extra server`）`uv run steproute serve` 后，浏览器打开 `http://127.0.0.1:8000` 能正常显示

常用环境变量（前缀 `STEPROUTE_`）：`LLM_API_KEY`、`LLM_BASE_URL`、`PROXY_URL`、`BACKEND`（`api`/`sim`）、`DATA_DIR`、`CONFIG_DIR`、`ARTIFACTS_DIR`、`LOG_LEVEL`。

---

## 🚀 使用

### 离线体验（不花钱、不联网）

把 `--data-dir examples/sample_corpus` 换成你自己的语料目录，就是真实数据了。

```powershell
uv run steproute ingest --data-dir examples/sample_corpus
uv run steproute search "baseline 对比 公平" --data-dir examples/sample_corpus
uv run steproute run --backend sim --data-dir examples/sample_corpus
```

### 真实调用

```powershell
uv run steproute run --backend api                          # 默认 step_level 路由
uv run steproute run --backend api -t "把 EXP-002 和 baseline 的对比结论重新核对一遍"
uv run steproute run --backend api --strategy always_strong # 对照策略：全程强模型
uv run steproute evaluate --backend api                     # 跑一次并给报告质量打分
```

`--strategy` 可选：`step_level`（默认）／`query_level`（开头判一次、全程锁定）／`always_strong`（全强，基线）／`always_fast`（全弱，基线）。

### 可视化网站

```powershell
uv sync --extra server
uv run steproute serve                    # 打开 http://127.0.0.1:8000
# 想快速看界面，直接用示例语料启动：
uv run steproute serve --data-dir examples/sample_corpus
```

网站有两个页签：**工作台**（选知识库、每日简报、待办清单、项目长期记忆）和**周报**（预览、生成、发布到飞书）。

也提供 Gradio 面板（可选）：

```powershell
uv sync --extra ui
uv run steproute ui                       # 默认 http://127.0.0.1:7860
```

### 命令一览

| 命令 | 作用 |
| --- | --- |
| `run` | 跑一次 Agent；`-t/--task` 指定任务，`-s/--strategy` 选策略，`-b/--backend` 选 `api`/`sim`，`--out` 导出完整 JSON，`--quiet` 只看报告 |
| `evaluate` | 真实/仿真运行 + rubric 质量评分 |
| `ingest` | 查看语料加载结果与抽取到的待办 |
| `search` | BM25 检索调试（positional 参传检索词，`--top-k` 控制条数） |
| `sync` | 从飞书知识库 / 云空间同步文档（含评论）到本地语料 |
| `brief` | 生成每日工作简报 |
| `memory` | 长期记忆：`show` / `state` / `decision` / `lesson` / `distill` |
| `weekly` | 周报：`preview`（本地预览）/ `publish`（写入飞书知识库） |
| `serve` | 启动可视化网站（需 `--extra server`） |
| `ui` | 启动 Web UI（需 `--extra ui`） |
| `secrets` | 密钥管理：`set` / `status` / `delete`，`--name llm_api_key\|feishu_app_secret` |
| `version` | 显示版本 |

### 飞书同步 / 简报 / 周报 / 长期记忆

```powershell
# 1) 从飞书知识库递归同步文档（先 --dry-run 只列目录、不落盘）
uv run steproute sync --wiki-space "<space_id>" --into meeting_minutes --dry-run
uv run steproute sync --wiki-space "<space_id>" --into meeting_minutes

# 2) 每日简报 与 长期记忆
uv run steproute brief
uv run steproute memory state --project "视频理解" --status 进行中 --goal "本周跑通公平对比"
uv run steproute memory distill                        # 从文档提炼「状态 / 决策 / 经验」

# 3) 周报：先本地预览，再写入飞书知识库
uv run steproute weekly preview
uv run steproute weekly publish --space "<target_space_id>"
```

同步下来的语料，可以被 `ingest / search / run / brief / weekly` 直接复用，检索、待办抽取和路由调度都不用改。

### 换成自己的科研材料

把文件丢进 `data/sources/` 下对应的子目录即可：

- `weekly_reports/`、`meeting_minutes/`：`.md` 或 `.txt`，可选 YAML 头部 `title` / `date` / `author`；
- `experiment_logs/`：`.json`，字段 `experiment_id` / `name` / `config` / `metrics` / `notes`；
- `repo/`：任意 `.md` 或 `.json`。

## 📄 许可

本项目基于 MIT License 开源。