"""装配层：把配置、语料、模型池、后端、路由与调度组装成一个可运行的 Harness。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent.prompts import SYSTEM_PROMPT
from .agent.runtime import HarnessRuntime
from .agent.tools import ToolExecutor
from .config import AppConfig, Settings, load_app_config
from .corpus.loaders import load_corpus
from .corpus.model import Corpus
from .corpus.retriever import Retriever
from .evaluation.rubric import Reference, build_reference
from .llm.backends import APIBackend, Backend, SimBackend
from .llm.pool import ModelPool
from .logging_utils import get_logger, setup_logging
from .routing.cache import PrefixCacheTracker
from .routing.jev import JevRouter
from .routing.scheduler import RuntimeScheduler

logger = get_logger(__name__)


@dataclass
class HarnessBundle:
    """一次装配的全部组件。"""

    settings: Settings
    config: AppConfig
    corpus: Corpus
    retriever: Retriever
    pool: ModelPool
    backend: Backend
    cache: PrefixCacheTracker
    jev: JevRouter
    scheduler: RuntimeScheduler
    executor: ToolExecutor
    reference: Reference

    def runtime(self, strategy: str = "step_level") -> HarnessRuntime:
        """构造一个指定对照策略的运行时。"""
        return HarnessRuntime(
            config=self.config,
            settings=self.settings,
            pool=self.pool,
            backend=self.backend,
            corpus=self.corpus,
            retriever=self.retriever,
            jev=self.jev,
            scheduler=self.scheduler,
            cache=self.cache,
            executor=self.executor,
            strategy=strategy,
        )

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()


def build_harness(
    *,
    settings: Settings | None = None,
    config: AppConfig | None = None,
    backend_name: str | None = None,
    use_llm_judge: bool | None = None,
    config_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> HarnessBundle:
    """装配 Harness。

    ``backend_name``：``api``（真实调用）或 ``sim``（离线确定性仿真）。
    未指定时读取 ``Settings.backend``。
    """
    settings = settings or Settings()
    if config_dir is not None:
        settings = settings.model_copy(update={"config_dir": Path(config_dir)})
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": Path(data_dir)})

    app_config = config or load_app_config(settings.config_dir)
    setup_logging(settings.log_level, settings.log_file)

    corpus = load_corpus(settings.data_dir)
    retriever = Retriever(corpus)
    pool = ModelPool(app_config.models)

    resolved_backend = (backend_name or settings.backend or "api").lower()
    backend: Backend
    if resolved_backend == "sim":
        backend = SimBackend()
    else:
        backend = APIBackend(settings)
    logger.info("Harness 装配完成：backend=%s，模型池=%s", resolved_backend, [rt.id for rt in pool.all])

    cache = PrefixCacheTracker()
    if use_llm_judge is None:
        use_llm_judge = resolved_backend == "api"
    jev = JevRouter(app_config.jev, pool, backend=backend, use_llm_judge=use_llm_judge)
    scheduler = RuntimeScheduler(app_config.scheduler, pool, cache)
    executor = ToolExecutor(corpus, retriever, max_output_chars=app_config.agent.max_tool_chars)

    return HarnessBundle(
        settings=settings,
        config=app_config,
        corpus=corpus,
        retriever=retriever,
        pool=pool,
        backend=backend,
        cache=cache,
        jev=jev,
        scheduler=scheduler,
        executor=executor,
        reference=build_reference(corpus),
    )


__all__ = ["HarnessBundle", "build_harness", "SYSTEM_PROMPT"]