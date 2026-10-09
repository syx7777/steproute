"""pytest 公共夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

from steproute.config import Settings
from steproute.harness import build_harness

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 仓库自带的示例语料（已从 data/ 移到这里，避免污染运行时数据目录）。
SAMPLE_CORPUS = PROJECT_ROOT / "tests" / "fixtures" / "sample_corpus"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_corpus() -> Path:
    """示例语料根目录（其下含 ``sources/``）。"""
    return SAMPLE_CORPUS


@pytest.fixture()
def settings() -> Settings:
    """固定使用示例语料 + 项目 configs，且强制离线后端。"""
    return Settings(
        _env_file=None,
        data_dir=SAMPLE_CORPUS,
        config_dir=PROJECT_ROOT / "configs",
        backend="sim",
        log_level="WARNING",
    )


@pytest.fixture()
def bundle(settings: Settings):
    harness = build_harness(settings=settings, backend_name="sim", use_llm_judge=False)
    try:
        yield harness
    finally:
        harness.close()