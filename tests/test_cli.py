"""CLI 冒烟测试。"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from steproute.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "steproute" in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("run", "evaluate", "ingest", "search", "sync", "secrets"):
        assert command in result.stdout


def test_ingest_outputs_stats(sample_corpus) -> None:
    result = runner.invoke(app, ["ingest", "--data-dir", str(sample_corpus)])
    assert result.exit_code == 0
    assert "documents" in result.stdout


def test_search_finds_hits(sample_corpus) -> None:
    result = runner.invoke(app, ["search", "baseline 对比 公平", "--data-dir", str(sample_corpus)])
    assert result.exit_code == 0
    assert "baseline" in result.stdout or "纪要" in result.stdout


def test_run_offline_writes_json(tmp_path) -> None:
    out = tmp_path / "run.json"
    result = runner.invoke(app, ["run", "--backend", "sim", "--quiet", "--out", str(out)])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["strategy"] == "step_level"
    assert payload["steps"] > 0


def test_secrets_status_masks_key(monkeypatch) -> None:
    monkeypatch.setenv("STEPROUTE_LLM_API_KEY", "sk-status-abcdefghijklmnop")
    result = runner.invoke(app, ["secrets", "status"])
    assert result.exit_code == 0
    assert "sk-status-abcdefghijklmnop" not in result.stdout
    assert "env" in result.stdout