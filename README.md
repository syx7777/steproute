<div align="center">

# steproute

**用最合适的模型，跑好每一步。**

面向多步 Agent 的**步级动态模型路由与调度**：先判断每一步最低需要什么能力，再由 Runtime Scheduler 在够用的模型里挑出运行时最合适的那个，并叠加 Cache-aware 与 Failure-aware 机制。

[![Python](https://img.shields.io/badge/python-3.11%20~%203.13-3776AB?logo=python\&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-2EA043.svg)](#-许可)
[![uv](https://img.shields.io/badge/packaged%20with-uv-5C5CFF.svg)](https://docs.astral.sh/uv/)
[![Typer](https://img.shields.io/badge/CLI-Typer-009688.svg)](https://typer.tiangolo.com/)
[![FastAPI](https://img.shields.io/badge/server-FastAPI-009688?logo=fastapi\&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

***

## 🚀 快速开始

```bash
git clone https://github.com/syx7777/steproute.git
cd steproute

uv sync --extra server                                                # 1) 安装依赖（含可视化网站所需依赖）

uv run steproute run --backend sim --data-dir examples/sample_corpus  # 2) 离线跑通完整 Agent（不联网、零成本）

uv run steproute serve                                                # 3) 启动可视化网站 → http://127.0.0.1:8000
```

> 步骤 1～2 不需要任何 API Key，也不产生费用。需要真实调用模型时再配置密钥，见「安装与配置」。

***

## 🛠️ 安装与配置

### 环境要求

| 项目                               | 要求                               |
| -------------------------------- | -------------------------------- |
| 操作系统                             | Windows / Linux / macOS          |
| Python                           | 3.11 ～ 3.13，建议 3.12（由 `uv` 自动管理） |
| [uv](https://docs.astral.sh/uv/) | **必需**，用于环境与依赖管理                 |

未安装 `uv` 时先安装：

```powershell
# Windows（PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. 安装依赖

```bash
uv sync                  # 核心依赖（Agent / CLI）
uv sync --extra server   # 追加 fastapi、uvicorn（可视化网站）
uv sync --extra ui       # 追加 gradio（Web UI）
```

> 只跑通流程，`uv sync` 就够了；要用网站再加 `--extra server`。

### 2. 验证部署

项目自带示例语料（`examples/sample_corpus/`，纯虚构数据），因不在默认 `data/` 目录下，命令需显式加 `--data-dir`：

```bash
uv run steproute version
uv run steproute ingest --data-dir examples/sample_corpus
uv run steproute search "baseline 对比 公平" --data-dir examples/sample_corpus
uv run steproute run --backend sim --data-dir examples/sample_corpus
```

`run` 结尾会打印运行结果，出现下面的「档位分布」即表示部署成功：

```text
───────────────────────── 运行结果 · 策略=step_level ──────────────────────────
    步数  7
    耗时  0.00 s
    成本  $0.003277
档位分布  {"fast": 3, "strong": 4}
    成功  是
```

档位分布反映按步判档的结果：检索、读取类步骤走 `fast`，`compare` 这类需要推理的步骤走 `strong`。

### 3. 配置密钥

密钥默认不落明文，解析顺序为：显式配置（`.env` / 环境变量）→ 操作系统凭据管理器 → 本地加密密钥库。

```powershell
# 推荐：写入系统凭据管理器（Windows 走 DPAPI 加密）
uv run steproute secrets set
uv run steproute secrets status              # 只看解析来源，显示掩码

# 或：仅对当前终端生效
$env:STEPROUTE_LLM_API_KEY = "<your-key>"

# 或：复制模板，填入 .env（.env 已在 .gitignore 中，不会入库）
Copy-Item .env.example .env
```

### 4. 常用环境变量

变量前缀统一为 `STEPROUTE_`：

| 变量              | 默认值                           | 说明                      |
| --------------- | ----------------------------- | ----------------------- |
| `LLM_API_KEY`   | 空                             | LLM 密钥                  |
| `LLM_BASE_URL`  | `https://api.deepseek.com/v1` | OpenAI 兼容端点             |
| `BACKEND`       | `api`                         | `api` 真实调用 / `sim` 离线仿真 |
| `PROXY_URL`     | 空                             | 出站代理，留空则直连              |
| `DATA_DIR`      | `data`                        | 语料根目录                   |
| `CONFIG_DIR`    | `configs`                     | 配置目录                    |
| `ARTIFACTS_DIR` | `artifacts`                   | 运行产物目录                  |
| `LOG_LEVEL`     | `INFO`                        | 日志级别                    |

### 5. 更换模型（可选）

默认按能力档映射到 DeepSeek 官方网关，配置在 [configs/models.yaml](configs/models.yaml)。切换为本地 vLLM 只需改这一个文件，路由与调度代码无需改动：

```yaml
models:
  - id: qwen-fast
    tier: fast
    model: Qwen2.5-1.5B-Instruct
    endpoint: http://127.0.0.1:8000/v1
    capability: 0.45
  - id: qwen-strong
    tier: strong
    model: Qwen2.5-32B-Instruct
    endpoint: http://127.0.0.1:8001/v1
    max_concurrency: 2      # 供调度器估计 GPU 队列
    capability: 0.93
```

***

## 📦 使用指南

### 命令一览

| 命令         | 作用                                                                                            |
| ---------- | --------------------------------------------------------------------------------------------- |
| `run`      | 跑一次 Agent；`-t/--task` 指定任务，`-s/--strategy` 选策略，`-b/--backend` 选 `api`/`sim`，`--out` 导出完整 JSON |
| `evaluate` | 真实或仿真运行 + rubric 质量评分                                                                         |
| `ingest`   | 查看语料加载结果与抽取到的待办                                                                               |
| `search`   | BM25 检索调试（位置参数为检索词，`--top-k` 控制条数）                                                            |
| `sync`     | 从飞书知识库 / 云空间同步文档（含评论）到本地语料                                                                    |
| `brief`    | 生成每日工作简报                                                                                      |
| `memory`   | 长期记忆：`show` / `state` / `decision` / `lesson` / `distill`                                     |
| `weekly`   | 周报：`preview`（本地预览）/ `publish`（写入飞书知识库）                                                        |
| `serve`    | 启动可视化网站（需 `--extra server`）                                                                   |
| `ui`       | 启动 Web UI（需 `--extra ui`）                                                                     |
| `secrets`  | 密钥管理：`set` / `status` / `delete`，`--name llm_api_key\|feishu_app_secret`                      |
| `version`  | 显示版本                                                                                          |

### 典型用法

**① 离线体验（不联网、零成本）**

```powershell
uv run steproute run --backend sim --data-dir examples/sample_corpus
```

把 `--data-dir` 换成自己的语料目录，即为真实数据。

**② 真实调用**

```powershell
uv run steproute run --backend api                          # 默认 step_level 路由
uv run steproute run --backend api -t "把 EXP-002 和 baseline 的对比结论重新核对一遍"
uv run steproute run --backend api --strategy always_strong # 对照策略：全程强模型
uv run steproute evaluate --backend api                     # 跑一次并给报告质量打分
```

`--strategy` 可选：`step_level`（默认）/ `query_level`（开头判一次、全程锁定）/ `always_strong`（全强，基线）/ `always_fast`（全弱，基线）。

**③ 可视化网站**

```powershell
uv sync --extra server
uv run steproute serve                    # 打开 http://127.0.0.1:8000
uv run steproute serve --data-dir examples/sample_corpus   # 用示例语料快速看界面
```

### 飞书同步、简报与周报

```powershell
# 1) 从飞书知识库递归同步文档（先 --dry-run 只列目录、不落盘）
uv run steproute sync --wiki-space "<space_id>" --into meeting_minutes --dry-run
uv run steproute sync --wiki-space "<space_id>" --into meeting_minutes

# 2) 每日简报与长期记忆
uv run steproute brief
uv run steproute memory state --project "视频理解" --status 进行中 --goal "本周跑通公平对比"
uv run steproute memory distill                        # 从文档提炼「状态 / 决策 / 经验」

# 3) 周报：先本地预览，再写入飞书知识库
uv run steproute weekly preview
uv run steproute weekly publish --space "<target_space_id>"
```

同步下来的语料可被 `ingest / search / run / brief / weekly` 直接复用，检索、待办抽取与路由调度均无需改动。

### 接入自己的科研材料

把文件放入 `data/sources/` 下对应子目录即可：

- `weekly_reports/`、`meeting_minutes/`：`.md` 或 `.txt`，可选 YAML 头部 `title` / `date` / `author`；
- `experiment_logs/`：`.json`，字段 `experiment_id` / `name` / `config` / `metrics` / `notes`；
- `repo/`：任意 `.md` 或 `.json`。

***

## ✨ 项目功能

| 能力 | 说明 |
| --- | --- |
| 🔀 **步级动态模型路由** | 判档粒度是「每一步」而不是「整道题」，简单步骤不占用强模型 |
| 🧮 **Runtime Scheduler** | 只在能力达标的候选模型里比较延迟、GPU 队列、前缀缓存与成本，不会为省钱而把任务交给做不动的模型 |
| ⚡ **Cache-aware** | 优先复用已命中前缀缓存的模型，降低首字延迟 |
| 🛡️ **Failure-aware** | 连续工具失败或连续无进展时，自动升级到更强模型重新分析 |
| ⚙️ **配置驱动** | 模型池、路由权重、调度权重都在 `configs/*.yaml`；切换到本地 vLLM 只改 `configs/models.yaml` |
| 🧪 **离线可复现** | 自带确定性离线后端（`--backend sim`）与示例语料，不联网、零成本即可验证整条链路 |
| 🧰 **工作助手** | 飞书文档 / 评论同步、每日简报、可勾选待办、项目长期记忆、周报生成与发布 |
| 🖥️ **可视化网站** | FastAPI 单页前端（工作台 / 周报），另提供可选的 Gradio 面板 |

判档默认走纯启发式（`heuristic`），不调用任何模型；`llm` 模式每步用最便宜的模型判一次；`hybrid` 模式仅在分数落入阈值临界区时兜底判档。

***

## 🎯 适用场景

**适合**

- 做**多步 Agent / 工作流**，一条任务要调用十几次以上模型，且各步难度差异明显；
- 希望在**质量与成本之间可控**，而不是「要么全强、要么全弱」；
- 打算接入**本地 vLLM 的多档模型**，想让调度跟随 GPU 队列、延迟、缓存动态选型；
- 有**散落在多处的科研或项目材料**（周报评论、组会纪要、实验记录、代码仓库），需要自动整理为进展与下一步待办。

**可能不太合适**

- 单轮问答、单步任务——只调用一次模型，没有「逐步路由」的空间；
- 只使用单一模型，且不在意成本与延迟的场景。

***

## 📁 项目结构

```text
steproute/
├── configs/                 # 全部可调参数（无需改代码）
│   ├── models.yaml          #   模型池：能力档 / 端点 / 成本 / 延迟画像 ← 换 vLLM 只改这里
│   ├── jev.yaml             #   判档：模式、阈值、特征权重、升级策略
│   ├── scheduler.yaml       #   调度器：延迟 / 成本 / 队列 / 缓存权重
│   ├── agent.yaml           #   Agent：最大步数、历史窗口
│   └── knowledge_bases.example.yaml   # 知识库列表模板
├── data/sources/            # 语料目录（可替换为自己的材料）
├── src/steproute/
│   ├── cli.py               # 命令行入口
│   ├── harness.py           # 装配层：把全部组件拼起来
│   ├── llm/                 # 模型池 + 后端（真实 API / 离线 Sim）
│   ├── corpus/              # 语料加载 + 中英文 BM25 检索
│   ├── connectors/          # 飞书同步（落盘式）
│   ├── agent/               # 状态 / 工具 / 主循环
│   ├── routing/             # 特征 / 判档 / 调度器 / 缓存追踪
│   ├── assistant/           # 简报 / 待办 / 长期记忆 / 周报
│   └── server/              # FastAPI 后端 + 单页前端
└── examples/                # 示例语料（纯虚构数据，可直接试跑）
```

***

## 📄 许可

本项目基于 **MIT License** 开源。
