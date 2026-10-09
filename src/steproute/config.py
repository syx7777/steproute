"""配置体系（三层）。

* :class:`Settings` —— 运行期环境配置（环境变量 / ``.env``，全部为**非敏感**项）。
* ``configs/models.yaml`` —— 模型池：能力档、端点、成本与调度画像。
* 其余 ``configs/*.yaml`` —— 路由（Jev）、调度（Scheduler）与 Agent 行为配置。

密钥不在这里解析，统一由 :mod:`steproute.secrets` 负责。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CapabilityTier = Literal["fast", "balanced", "strong"]

#: 能力档的强弱顺序，供调度器比较与"升级"使用。
TIER_ORDER: dict[str, int] = {"fast": 0, "balanced": 1, "strong": 2}


def tier_at_least(tier: str, floor: str) -> bool:
    """``tier`` 的能力是否不低于 ``floor``。"""
    return TIER_ORDER.get(tier, 0) >= TIER_ORDER.get(floor, 0)


def tier_rank(tier: str) -> int:
    return TIER_ORDER.get(tier, 0)


# --------------------------------------------------------------------------- #
# 运行期设置
# --------------------------------------------------------------------------- #

class Settings(BaseSettings):
    """环境变量驱动的运行期设置（非敏感）。"""

    model_config = SettingsConfigDict(
        env_prefix="STEPROUTE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM 网关 ----
    #: LLM API Key。**可选**：为"把凭证都放进 .env"的本地便利而设；
    #: 填写即为**明文落盘**，安全性低于 secrets（keyring / 加密密钥库）。
    #: 留空时回退到 :mod:`steproute.secrets` 解析链。``repr=False`` 避免打印时泄漏。
    llm_api_key: str = Field(default="", repr=False)
    #: 默认网关：既可以是 DeepSeek 官方，也可以是本地 vLLM 的 OpenAI 兼容地址。
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_timeout: float = 180.0
    llm_max_retries: int = 3
    llm_temperature: float = 0.2
    llm_max_tokens: int = 4096
    request_delay: float = 0.0
    #: 是否启用"仿真后端"（不发起真实网络请求，用于离线指标准与消融实验）。
    backend: Literal["api", "sim"] = "api"

    # ---- 网络 ----
    #: 代理为**可选**能力：默认 None（直连）。
    #: 只有当确实需要访问外网且直连不通时，才设置
    #: STEPROUTE_PROXY_URL=http://127.0.0.1:7892 或使用 --proxy 参数。
    proxy_url: str | None = None
    trust_env_proxy: bool = True
    http_timeout: float = 30.0

    # ---- 外部来源同步（飞书）----
    #: 飞书开放平台基地址（仅 ``steproute sync`` 使用）。
    feishu_base_url: str = "https://open.feishu.cn/open-apis"
    #: 飞书自建应用 App ID（非敏感项）。
    feishu_app_id: str = ""
    #: 飞书自建应用 App Secret。**可选**：仅为"把凭证都放进 .env"的本地便利而设；
    #: 一旦填写即为**明文落盘**，安全性低于 secrets（keyring / 加密密钥库）。
    #: 留空时回退到 :mod:`steproute.secrets` 解析链。``repr=False`` 避免打印时泄漏。
    feishu_app_secret: str = Field(default="", repr=False)
    #: 拼接文档链接用的站点地址。
    feishu_site_url: str = "https://feishu.cn"
    #: 周报写入的知识库 space_id（``steproute weekly publish`` 使用）。
    feishu_weekly_space_id: str = ""
    #: 周报模板文档 token；留空则按标题关键词在知识库内自动匹配。
    feishu_weekly_template: str = ""
    #: 自动匹配模板时使用的标题关键词。
    feishu_weekly_template_keyword: str = "模板"

    # ---- 运行时 ----
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    config_dir: Path = Path("configs")
    log_level: str = "INFO"
    log_file: Path | None = None

    def proxy_for_httpx(self) -> str | None:
        """返回需要显式使用的代理；为空字符串 / None 表示直连。"""
        value = (self.proxy_url or "").strip()
        return value or None

    def with_proxy(self, raw: str | None) -> Settings:
        """按 CLI 传入的原始取值覆盖代理设置。

        * ``raw is None`` —— 未指定，保持当前设置（可能是环境变量指定的代理）；
        * ``"" / none / off / direct`` —— 强制直连；
        * 其它取值 —— 视为代理 URL。
        """
        if raw is None:
            return self
        from .net import parse_proxy_option

        return self.model_copy(update={"proxy_url": parse_proxy_option(raw)})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# 模型池
# --------------------------------------------------------------------------- #

class ModelSpecConfig(BaseModel):
    """池中单个模型的静态画像。"""

    id: str
    tier: CapabilityTier
    model: str
    #: None 表示沿用 ``Settings.llm_base_url``（便于用 vLLM 本地端点整体替换）。
    endpoint: str | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.2
    #: 单价（美元 / 百万 token），用于成本核算。
    price_prompt_per_m: float = 0.0
    price_completion_per_m: float = 0.0
    #: 调度画像：并发上限、空载延迟、随 token 增长的延迟斜率（秒）。
    max_concurrency: int = 4
    base_latency_s: float = 1.0
    latency_per_1k_prompt_s: float = 0.015
    latency_per_1k_completion_s: float = 0.06
    #: 仿真画像：0~1 的能力值 & 是否支持前缀缓存。
    capability: float = 0.5
    supports_cache: bool = True


class ModelPoolConfig(BaseModel):
    models: list[ModelSpecConfig]

    @model_validator(mode="after")
    def _ensure_unique_ids(self) -> ModelPoolConfig:
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("models.yaml 中存在重复的模型 id")
        if not self.models:
            raise ValueError("models.yaml 至少要定义一个模型")
        return self

    def by_tier(self, tier: str) -> list[ModelSpecConfig]:
        return [m for m in self.models if m.tier == tier]

    def get(self, model_id: str) -> ModelSpecConfig:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"未知模型 id: {model_id}")


# --------------------------------------------------------------------------- #
# 路由（Jev）与调度（Scheduler）
# --------------------------------------------------------------------------- #

class EscalationConfig(BaseModel):
    """Failure-aware：失败 / 无进展时的能力升级策略。"""

    enabled: bool = True
    #: 连续失败达到该次数即升级一档。
    consecutive_failures: int = 2
    #: 连续无进展（没有新增证据 / 子目标未推进）达到该次数即升级一档。
    no_progress_streak: int = 2
    #: 单次升级跨越的档位数上限。
    max_bump: int = 1


class StepClassifierConfig(BaseModel):
    """把"当前步骤类型"判为 fast/balanced/strong 的规则表。"""

    #: 各步骤类型的基础难度（0~1）。
    base_difficulty: dict[str, float] = Field(
        default_factory=lambda: {
            "retrieve": 0.10,
            "read": 0.12,
            "extract": 0.18,
            "compare": 0.55,
            "analyze": 0.72,
            "synthesize": 0.60,
            "final": 0.65,
        }
    )
    #: 各步骤类型的能力下限（硬约束）。
    min_tier: dict[str, CapabilityTier] = Field(
        default_factory=lambda: {
            "retrieve": "fast",
            "read": "fast",
            "extract": "fast",
            "compare": "balanced",
            "analyze": "strong",
            "synthesize": "balanced",
            "final": "balanced",
        }
    )


class JevConfig(BaseModel):
    """Jev：判断"当前步骤最低需要什么能力"。

    ``mode``：

    * ``heuristic`` —— 纯特征加权打分（零额外调用，最快）；
    * ``llm``       —— 用最便宜模型做一次结构化判断（更准，但有额外开销）；
    * ``hybrid``    —— 先用启发式打分；只有在"临界区"（分数落在阈值附近）时才调用 LLM 兜底。
    """

    mode: Literal["heuristic", "llm", "hybrid"] = "hybrid"
    #: 难度分 → 能力档的阈值：<t0 → fast；<t1 → balanced；否则 strong。
    tier_thresholds: tuple[float, float] = (0.34, 0.66)
    step_classifier: StepClassifierConfig = Field(default_factory=StepClassifierConfig)
    #: 各特征的权重（会乘到 0~1 的归一化特征上）。
    feature_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "context_len": 0.06,
            "depth": 0.06,
            "failures": 0.08,
            "no_progress": 0.08,
            "evidence_gap": 0.10,
            "multi_source": 0.08,
            "numeric_conflict": 0.18,
            "ambiguity": 0.14,
            "staleness": 0.08,
        }
    )
    #: hybrid 模式下，难度分靠近阈值 ±band 时触发 LLM 兜底。
    hybrid_band: float = 0.08
    #: LLM 兜底使用的能力档（用最便宜的档即可）。
    llm_judge_tier: CapabilityTier = "fast"
    llm_judge_max_tokens: int = 384
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)


class SchedulerConfig(BaseModel):
    """Runtime Scheduler：在满足能力下限的候选模型中挑"运行时最合适"的一个。"""

    w_latency: float = 1.0
    w_cost: float = 0.35
    w_queue: float = 0.60
    w_cache: float = 0.55
    #: 命中前缀缓存可抵扣的等效延迟（秒）。
    cache_bonus_s: float = 0.9
    #: 每个在途请求的等效延迟惩罚（秒）。
    queue_penalty_s: float = 0.7
    #: 成本缩放（把"美元"折算成与"秒"可比的量纲）。
    cost_scale: float = 1.0
    #: 是否允许选择比 Jev 下限更强的档（默认允许，由打分决定）。
    allow_stronger: bool = True
    #: 是否冻结"能力恰好达标"的档位（禁止无谓升级到 strong）。
    freeze_strong_unless_required: bool = True


# --------------------------------------------------------------------------- #
# Agent 行为
# --------------------------------------------------------------------------- #

class AgentConfig(BaseModel):
    """Agent 行为配置。"""

    max_steps: int = 14
    #: 上下文里保留的历史步数（更早的被压缩成摘要）。
    history_window: int = 6
    #: 单篇文档注入 prompt 的最大字符数。
    max_doc_chars: int = 4000
    #: 单步工具输出注入 prompt 的最大字符数。
    max_tool_chars: int = 3000
    #: 检索返回的片段数量。
    retrieval_top_k: int = 5
    #: 达到该步数仍无证据时，强制进入综合分析。
    force_synthesis_after: int = 8


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #

class AppConfig(BaseModel):
    models: ModelPoolConfig
    jev: JevConfig = Field(default_factory=JevConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


# --------------------------------------------------------------------------- #
# 默认值（配置文件缺失时的兜底）
# --------------------------------------------------------------------------- #

def default_model_pool() -> ModelPoolConfig:
    """内置模型池：DeepSeek 官方按能力档映射；换成 vLLM 时只改 endpoint 即可。"""
    return ModelPoolConfig(
        models=[
            ModelSpecConfig(
                id="flash-fast",
                tier="fast",
                model="deepseek-flash",
                endpoint=None,
                reasoning_effort="low",
                max_tokens=2048,
                price_prompt_per_m=0.14,
                price_completion_per_m=0.28,
                max_concurrency=8,
                base_latency_s=0.9,
                capability=0.45,
            ),
            ModelSpecConfig(
                id="flash-balanced",
                tier="balanced",
                model="deepseek-flash",
                endpoint=None,
                reasoning_effort="high",
                max_tokens=4096,
                price_prompt_per_m=0.14,
                price_completion_per_m=0.28,
                max_concurrency=6,
                base_latency_s=1.4,
                capability=0.72,
            ),
            ModelSpecConfig(
                id="pro-strong",
                tier="strong",
                model="deepseek-v4-pro",
                endpoint=None,
                reasoning_effort="high",
                max_tokens=6144,
                price_prompt_per_m=0.55,
                price_completion_per_m=2.19,
                max_concurrency=3,
                base_latency_s=2.6,
                capability=0.93,
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #

def load_yaml(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_app_config(config_dir: Path | str = "configs") -> AppConfig:
    """从配置目录加载 YAML（缺省文件自动回退到内置默认值）。"""
    directory = Path(config_dir)
    models_raw = load_yaml(directory / "models.yaml")
    models = ModelPoolConfig(**models_raw) if models_raw else default_model_pool()
    return AppConfig(
        models=models,
        jev=JevConfig(**load_yaml(directory / "jev.yaml")),
        scheduler=SchedulerConfig(**load_yaml(directory / "scheduler.yaml")),
        agent=AgentConfig(**load_yaml(directory / "agent.yaml")),
    )


# --------------------------------------------------------------------------- #
# 知识库（简报页选择器 + 周报固定写入目标）
# --------------------------------------------------------------------------- #

#: ``configs/knowledge_bases.yaml`` 缺失时的兜底。
DEFAULT_WEEKLY_SPACE_ID = "7671654639892171748"
DEFAULT_KNOWLEDGE_BASES: tuple[tuple[str, str], ...] = (
    ("实习", "7552806097184784386"),
    ("风控项目", "7672220327107103924"),
    ("AI4Phy", "7552812149661057025"),
    ("自学", "7626974321562389714"),
)


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    """可选知识库列表 + 周报固定写入的知识库。"""

    weekly_space_id: str
    options: tuple[tuple[str, str], ...]

    def to_list(self) -> list[dict[str, str]]:
        return [{"name": name, "space_id": space_id} for name, space_id in self.options]

    def name_of(self, space_id: str) -> str:
        for name, candidate in self.options:
            if candidate == space_id:
                return name
        return space_id


def load_knowledge_bases(config_dir: Path | str = "configs") -> KnowledgeBaseConfig:
    """读取 ``configs/knowledge_bases.yaml``（缺失或为空时回退内置默认值）。"""
    raw = load_yaml(Path(config_dir) / "knowledge_bases.yaml")
    weekly = str(raw.get("weekly_space_id") or DEFAULT_WEEKLY_SPACE_ID)

    options: list[tuple[str, str]] = []
    for item in raw.get("knowledge_bases") or []:
        if isinstance(item, dict) and item.get("name") and item.get("space_id"):
            options.append((str(item["name"]), str(item["space_id"])))
    return KnowledgeBaseConfig(weekly_space_id=weekly, options=tuple(options) or DEFAULT_KNOWLEDGE_BASES)