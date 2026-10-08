"""pytest 公共夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

from steproute.config import Settings
from steproute.harness import build_harness

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def settings() -> Settings:
    """固定使用项目根目录下的 data / configs，且强制离线后端。"""
    return Settings(
        data_dir=PROJECT_ROOT / "data",
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