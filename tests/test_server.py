"""可视化网站后端测试（需要可选依赖 fastapi；未安装时自动跳过）。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from steproute.assistant.distill import build_distill_backend  # noqa: E402
from steproute.assistant.memory import MemoryStore  # noqa: E402
from steproute.assistant.summarize import BriefSummary, save_summary  # noqa: E402
from steproute.config import Settings  # noqa: E402
from steproute.llm.schema import ChatResult  # noqa: E402
from steproute.server import app as app_module  # noqa: E402
from steproute.server.app import create_app  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
#: 占位知识库配置（测试自建，避免依赖开发者本地的 configs/knowledge_bases.yaml）。
KB_NAME = "项目A"
WEEKLY_SPACE_ID = "0000000000000000000"
SPACE_ID = "1111111111111111111"  # 项目A
OTHER_SPACE_ID = "2222222222222222222"  # 项目B
#: 文档待办只取「本周更新过」的文档，夹具日期用今天以保证落在当前自然周内。
TODAY = date.today().isoformat()


class _FakeLLMBackend:
    """返回固定 JSON 的假后端：离线、不发网络请求，并记录调用次数。"""

    name = "fake"

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0
        self.messages: list[list[dict[str, str]]] = []

    def complete(self, spec, messages, **kwargs) -> ChatResult:
        self.calls += 1
        self.messages.append(list(messages))
        return ChatResult(content=self.content)


class _FakeSyncResult:
    def to_dict(self) -> dict[str, int]:
        return {"written": 0, "skipped": 0}


class _FakeFeishuClient:
    def __init__(self, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _balanced_spec():
    settings = Settings(_env_file=None, config_dir=PROJECT_ROOT / "configs", backend="sim", log_level="WARNING")
    _, spec = build_distill_backend(settings)
    return spec

def _notes_doc() -> str:
    return f"""---
title: 实习笔记
date: {TODAY}
---

结论：EXP-002 与 baseline 配置不一致。
教训：代理不稳定时不要用大模型下载。
- 下一步：补齐实习周报模板。
"""


def _client(tmp_path: Path) -> TestClient:
    data = tmp_path / "data"
    minutes_dir = data / "sources" / "meeting_minutes"
    minutes_dir.mkdir(parents=True)
    (minutes_dir / f"{TODAY}_group.md").write_text(
        f"---\ntitle: 组会纪要\ndate: {TODAY}\n---\n\n- 下一步：统一抽帧数后重跑\n",
        encoding="utf-8",
    )
    notes_dir = data / "sources" / "notes" / SPACE_ID
    notes_dir.mkdir(parents=True)
    (notes_dir / "实习笔记.md").write_text(_notes_doc(), encoding="utf-8")

    settings = Settings(
        _env_file=None,
        data_dir=data,
        config_dir=_write_kb_config(tmp_path),
        log_level="WARNING",
    )
    return TestClient(create_app(settings))


def _write_kb_config(tmp_path: Path) -> Path:
    """在临时目录写一份占位知识库配置，返回可作为 config_dir 的目录。

    其余配置文件（models/jev/scheduler/agent）缺失时会自动回退内置默认值。
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "knowledge_bases.yaml").write_text(
        f'weekly_space_id: "{WEEKLY_SPACE_ID}"\n'
        "knowledge_bases:\n"
        f'  - {{name: {KB_NAME}, space_id: "{SPACE_ID}"}}\n'
        f'  - {{name: 项目B, space_id: "{OTHER_SPACE_ID}"}}\n',
        encoding="utf-8",
    )
    return config_dir


def test_dashboard_returns_all_sections(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/api/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {"date", "brief", "todos", "memory", "weekly", "knowledge_bases", "weekly_space_id"}
    assert payload["weekly_space_id"] == WEEKLY_SPACE_ID
    assert payload["todo_stats"]["total"] <= 10


def test_dashboard_scopes_todos_to_selected_space(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        unscoped = client.get("/api/dashboard").json()
        scoped = client.get("/api/dashboard", params={"space": SPACE_ID}).json()

    assert any("统一抽帧数" in item["text"] for item in unscoped["todos"])  # 会议纪要里的
    scoped_texts = [item["text"] for item in scoped["todos"]]
    assert all("统一抽帧数" not in text for text in scoped_texts)  # 被知识库过滤掉
    assert any("补齐实习周报模板" in text for text in scoped_texts)  # 该知识库里的
    assert scoped["space_id"] == SPACE_ID


def test_dashboard_memory_is_scoped_by_space(tmp_path: Path) -> None:
    """记忆面板按「知识库归属」过滤：只显示该库 + 无归属条目，不含只属于其它库的条目。"""
    with _client(tmp_path) as client:
        memory = MemoryStore(tmp_path / "data")
        memory.add_decision("实习库的决策", space_id=SPACE_ID)
        memory.add_decision("风控库的决策", space_id=OTHER_SPACE_ID)
        memory.add_decision("无归属的手写决策")

        scoped = client.get("/api/dashboard", params={"space": SPACE_ID}).json()
        texts = [entry["text"] for entry in scoped["memory"]["decisions"]]
        assert "实习库的决策" in texts
        assert "无归属的手写决策" in texts  # 无归属条目各库可见
        assert "风控库的决策" not in texts  # 只属于别的知识库
        assert scoped["memory"]["stats"]["decisions"] == 2
        # 规则回退进展文案用知识库中文名（space_label=knowledge.name_of(space_id)）
        progress = scoped["brief"]["progress"]
        progress_text = "\n".join(progress) if isinstance(progress, list) else progress
        assert KB_NAME in progress_text  # 规则回退进展文案用知识库中文名（space_label=knowledge.name_of(space_id)）

        # 默认不过滤：全部可见
        unscoped = client.get("/api/dashboard").json()
        assert unscoped["memory"]["stats"]["decisions"] == 3

        # /api/memory 同样支持 space 过滤
        api_scoped = client.get("/api/memory", params={"space": SPACE_ID}).json()
        api_texts = [entry["text"] for entry in api_scoped["decisions"]]
        assert "风控库的决策" not in api_texts and "实习库的决策" in api_texts
        assert api_scoped["stats"]["decisions"] == 2
        assert api_scoped["space_id"] == SPACE_ID


def test_dashboard_has_summary_and_candidates(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        payload = client.get("/api/dashboard").json()
    assert "summary" in payload and "todo_candidates" in payload
    assert payload["todo_candidates"] >= 1
    assert isinstance(payload["candidate_todos"], list)
    # dashboard 只读缓存：还没归纳过时 summary 为空
    assert payload["summary"] is None


def test_dashboard_prefers_llm_todos_when_summary_available(tmp_path: Path) -> None:
    """归纳成功（缓存 method=llm）后，清单展示模型整理的待办（只读），替换文档待办。"""
    with _client(tmp_path) as client:
        save_summary(
            tmp_path / "data",
            BriefSummary(todos=["归纳出的待办 A", "归纳出的待办 B"], method="llm", space_id=SPACE_ID),
        )
        payload = client.get("/api/dashboard", params={"space": SPACE_ID}).json()
        llm_items = [item for item in payload["todos"] if item["source"] == "llm"]
        assert [item["text"] for item in llm_items] == ["归纳出的待办 A", "归纳出的待办 B"]
        assert all(item["source"] != "corpus" for item in payload["todos"])  # 文档待办被替换

        # /api/todos（勾选/编辑后的重载）与 dashboard 口径一致
        todos_api = client.get("/api/todos", params={"space": SPACE_ID}).json()
        assert [item["text"] for item in todos_api["todos"] if item["source"] == "llm"] == [
            "归纳出的待办 A",
            "归纳出的待办 B",
        ]


def test_knowledge_bases_list(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        payload = client.get("/api/knowledge-bases").json()
    assert [item["name"] for item in payload["knowledge_bases"]] == [KB_NAME, "项目B"]
    assert payload["weekly_space_id"] == WEEKLY_SPACE_ID


# --------------------------------------------------------------------------- #
# 待办 CRUD
# --------------------------------------------------------------------------- #

def test_manual_todo_crud(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post("/api/todos", json={"text": "写周报", "owner": "我"}).json()
        assert created["status"] == "open"
        todo_id = created["id"]

        assert client.put(f"/api/todos/manual/{todo_id}", json={"text": "写周报（含实验）"}).json()["text"] == "写周报（含实验）"
        assert client.put(f"/api/todos/manual/{todo_id}", json={"status": "done"}).json()["status"] == "done"
        assert any(item["id"] == todo_id for item in client.get("/api/todos").json()["todos"])

        assert client.delete(f"/api/todos/manual/{todo_id}").json()["ok"] is True
        assert client.delete(f"/api/todos/manual/{todo_id}").status_code == 404
        assert client.put("/api/todos/manual/nope", json={"text": "x"}).status_code == 404


def test_corpus_todo_can_be_toggled_edited_and_hidden(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        todos = client.get("/api/todos").json()["todos"]
        target = next(item for item in todos if item["source"] == "corpus")
        key = target["key"]

        # 勾选完成（写 override，不改源文档）
        client.put(f"/api/todos/corpus/{key}", json={"status": "done"})
        after = client.get("/api/todos").json()["todos"]
        assert next(item for item in after if item["key"] == key)["status"] == "done"

        # 改写文本
        client.put(f"/api/todos/corpus/{key}", json={"text": "改过的文档待办"})
        after = client.get("/api/todos").json()["todos"]
        assert next(item for item in after if item["key"] == key)["text"] == "改过的文档待办"

        # 删除（隐藏）
        assert client.delete(f"/api/todos/corpus/{key}").json()["ok"] is True
        assert all(item["key"] != key for item in client.get("/api/todos").json()["todos"])

        # 源文档未改动
        source = (tmp_path / "data" / "sources" / "meeting_minutes" / f"{TODAY}_group.md").read_text(encoding="utf-8")
        assert "统一抽帧数后重跑" in source


# --------------------------------------------------------------------------- #
# 长期记忆提炼
# --------------------------------------------------------------------------- #

def test_distill_falls_back_to_rules_without_llm(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/api/memory/distill", json={"space_id": SPACE_ID, "use_llm": False})
    assert response.status_code == 200
    payload = response.json()
    assert payload["method"] in {"rules", "none"}
    assert payload["docs"] >= 1
    if payload["method"] == "rules":
        assert any("配置不一致" in item for item in payload["decisions"])
        assert any("代理不稳定" in item for item in payload["lessons"])
        # 提炼结果写进了长期记忆
        assert payload["dashboard"]["memory"]["stats"]["decisions"] >= 1


# --------------------------------------------------------------------------- #
# 简报 / 待办归纳
# --------------------------------------------------------------------------- #

def test_brief_summarize_falls_back_to_rules_without_llm(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/api/brief/summarize", json={"space_id": SPACE_ID, "use_llm": False})
        assert response.status_code == 200
        payload = response.json()
        assert payload["method"] == "rules"  # 无 API Key → 回退规则（不发起网络请求）
        assert payload["todos"]
        assert payload["docs"] >= 1  # 本周文档数已统计（该知识库今天的笔记）
        assert "dashboard" in payload
        # 归纳结果已落缓存，dashboard 能读到
        assert payload["dashboard"]["summary"]["method"] == "rules"

        again = client.get("/api/dashboard", params={"space": SPACE_ID}).json()
    assert again["summary"] is not None
    assert again["summary"]["space_id"] == SPACE_ID


def test_todos_endpoint_reports_candidates(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        payload = client.get("/api/todos").json()
    assert payload["candidates"] >= 1
    assert isinstance(payload["candidate_todos"], list)


def test_brief_refresh_uses_injected_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """/api/brief/refresh 必须真的把后端传给 summarize_brief（不能静默走规则）。"""
    backend = _FakeLLMBackend('{"todos": ["重跑公平对比"], "progress": "项目正在进行公平对比。"}')
    monkeypatch.setattr(app_module, "FeishuClient", _FakeFeishuClient)
    monkeypatch.setattr(app_module, "sync_source", lambda *args, **kwargs: _FakeSyncResult())
    monkeypatch.setattr(app_module, "build_distill_backend", lambda settings: (backend, _balanced_spec()))

    with _client(tmp_path) as client:
        response = client.post("/api/brief/refresh", json={"space_id": SPACE_ID, "distill": True})

    assert response.status_code == 200
    payload = response.json()
    assert backend.calls >= 1  # 确实调用了注入的后端
    assert payload["summary"]["method"] == "llm"  # 而不是规则回退
    assert payload["summary"]["progress"] == "项目正在进行公平对比。"
    assert payload["summary"]["docs"] >= 1  # 本周文档数
    user_message = backend.messages[-1][-1]["content"]
    assert "EXP-002 与 baseline 配置不一致" in user_message  # 直读该知识库本周文档正文


def test_manual_todo_space_isolation_via_api(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        created = client.post("/api/todos", json={"text": "风控待办", "space_id": OTHER_SPACE_ID}).json()
        assert created["space_id"] == OTHER_SPACE_ID

        in_other = client.get("/api/todos", params={"space": OTHER_SPACE_ID}).json()["todos"]
        assert any(item["text"] == "风控待办" for item in in_other)

        in_sp = client.get("/api/todos", params={"space": SPACE_ID}).json()["todos"]
        assert all(item["text"] != "风控待办" for item in in_sp)  # 其它知识库看不到

        unscoped = client.get("/api/todos").json()["todos"]
        assert any(item["text"] == "风控待办" for item in unscoped)  # 不指定 space 时可见


# --------------------------------------------------------------------------- #
# 周报 / 页面
# --------------------------------------------------------------------------- #

def test_weekly_preview_and_publish_error_without_credentials(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        preview = client.get("/api/weekly", params={"day": "2026-10-05"})
        assert preview.status_code == 200
        assert "本周进展" in preview.json()["markdown"]
        assert preview.json()["weekly_space_id"] == WEEKLY_SPACE_ID

        publish = client.post("/api/weekly/publish", json={"date": "2026-10-05"})
        assert publish.status_code == 400
        assert "App ID" in publish.json()["detail"]


def test_weekly_summarize_falls_back_to_rules_without_llm(tmp_path: Path) -> None:
    """/api/weekly/summarize 按「本周 + 知识库」归纳；无 Key 时回退规则并写入缓存。"""
    with _client(tmp_path) as client:
        response = client.post("/api/weekly/summarize", json={"space_id": SPACE_ID, "day": TODAY, "use_llm": False})
        assert response.status_code == 200
        payload = response.json()
        assert payload["method"] == "rules"
        assert payload["docs"] == 1  # 该知识库今天的笔记
        assert "本周进展" in payload["markdown"]

        # 归纳结果已落缓存，预览能读回（按 space 隔离）
        preview = client.get("/api/weekly", params={"space": SPACE_ID, "day": TODAY}).json()
        assert preview["markdown"] == payload["markdown"]
        assert preview["space_id"] == SPACE_ID

        other = client.get("/api/weekly", params={"space": OTHER_SPACE_ID, "day": TODAY}).json()
        assert other["method"] == "none"  # 另一个知识库本周没有更新文档


def test_refresh_requires_credentials_not_silent(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/api/brief/refresh", json={"space_id": SPACE_ID})
    assert response.status_code == 400
    assert "App ID" in response.json()["detail"]


def test_index_page_served(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "科研助手" in response.text
    assert "工作台" in response.text and "周报" in response.text