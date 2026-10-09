"""飞书同步连接器的离线测试（httpx.MockTransport，不联网、不需要真实凭证）。"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from steproute.cli import app
from steproute.config import Settings
from steproute.connectors.base import RemoteDoc
from steproute.connectors.feishu import FEISHU_APP_SECRET_ENV, FeishuAuthError, FeishuClient
from steproute.connectors.sync import SyncTargetError, assign_stems, sync_source
from steproute.corpus.loaders import load_corpus
from steproute.secrets import EncryptedVault, SecretsManager

APP_ID = "cli_testapp00000000"
APP_SECRET = "test-app-secret-0000000000"
TENANT_TOKEN = "t-test-tenant-token"
SPACE_ID = "6946843325487906839"
FOLDER = "fldbcO1UuPz8VwnpPx5a9abcef"

# 时间戳对应 2025-09-10 / 2025-09-24（UTC）
WIKI_NODES: list[dict[str, object]] = [
    {
        "node_token": "wikA",
        "obj_token": "doxA1",
        "obj_type": "docx",
        "title": "组会纪要 2025-09-10",
        "obj_edit_time": "1757469600",
        "has_child": False,
    },
    {
        "node_token": "wikB",
        "obj_token": "doxB2",
        "obj_type": "docx",
        "title": "组会纪要 2025-09-24",
        "obj_edit_time": "1758679200",
        "has_child": False,
    },
    {"node_token": "wikC", "obj_token": "shtC3", "obj_type": "sheet", "title": "评测表", "has_child": False},
]

BODIES: dict[str, str] = {
    "doxA1": "组会纪要\n下一步：统一 baseline 抽帧数后重跑对比。",
    "doxB2": "组会纪要\n确认 EXP-002 与 baseline 分辨率不一致。",
    "fldDoc1": "云空间会议纪要\n待办：补齐基线复现的随机种子。",
    "fldDoc2": "云空间会议纪要二\n结论：暂不调整评测切分。",
}

COMMENTS: dict[str, list[dict[str, object]]] = {
    "doxA1": [
        {
            "comment_id": "c1",
            "reply_list": {
                "replies": [{"content": {"elements": [{"text_run": {"text": "导师：先对齐分辨率再比"}}]}}]
            },
        }
    ],
    "doxB2": [],
}

FOLDER_TREE: dict[str, list[dict[str, object]]] = {
    FOLDER: [
        {
            "name": "云空间会议纪要",
            "token": "fldDoc1",
            "type": "docx",
            "modified_time": "1757469600",
            "url": "https://feishu.cn/docx/fldDoc1",
        },
        {"name": "子文件夹", "token": "fldSub", "type": "folder"},
    ],
    "fldSub": [
        {"name": "云空间会议纪要二", "token": "fldDoc2", "type": "docx", "modified_time": "1758679200"}
    ],
}


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/open-apis/auth/v3/tenant_access_token/internal":
        body = json.loads(request.content.decode("utf-8") or "{}")
        if body.get("app_id") != APP_ID or body.get("app_secret") != APP_SECRET:
            return httpx.Response(200, json={"code": 10003, "msg": "app not found"})
        return httpx.Response(
            200,
            json={"code": 0, "msg": "ok", "tenant_access_token": TENANT_TOKEN, "expire": 7200},
        )

    if request.headers.get("Authorization") != f"Bearer {TENANT_TOKEN}":
        return httpx.Response(401, json={"code": 99991663, "msg": "invalid access token"})

    if path == f"/open-apis/wiki/v2/spaces/{SPACE_ID}/nodes":
        return httpx.Response(200, json={"code": 0, "msg": "ok", "data": {"items": WIKI_NODES, "has_more": False}})

    if path == "/open-apis/drive/v1/files":
        folder = request.url.params.get("folder_token") or ""
        return httpx.Response(
            200, json={"code": 0, "msg": "ok", "data": {"files": FOLDER_TREE.get(folder, []), "has_more": False}}
        )

    prefix = "/open-apis/docx/v1/documents/"
    if path.startswith(prefix) and path.endswith("/raw_content"):
        doc_id = path[len(prefix) : -len("/raw_content")]
        return httpx.Response(200, json={"code": 0, "msg": "ok", "data": {"content": BODIES.get(doc_id, "")}})

    prefix = "/open-apis/drive/v1/files/"
    if path.startswith(prefix) and path.endswith("/comments"):
        token = path[len(prefix) : -len("/comments")]
        return httpx.Response(
            200, json={"code": 0, "msg": "ok", "data": {"items": COMMENTS.get(token, []), "has_more": False}}
        )

    return httpx.Response(404, json={"code": 404, "msg": "not found"})


def _http() -> httpx.Client:
    return httpx.Client(base_url="https://open.feishu.cn/open-apis", transport=httpx.MockTransport(_handler))


def _feishu(tmp_path: Path, *, space_id: str | None = SPACE_ID, folder_token: str | None = None,
            app_id: str | None = APP_ID, app_secret: str | None = APP_SECRET) -> FeishuClient:
    return FeishuClient(
        space_id=space_id,
        folder_token=folder_token,
        app_id=app_id,
        app_secret=app_secret,
        settings=Settings(_env_file=None, data_dir=tmp_path, http_timeout=5.0, feishu_app_id=""),
        client=_http(),
    )


def test_list_documents_from_wiki_space(tmp_path: Path) -> None:
    client = _feishu(tmp_path)
    try:
        docs = client.list_documents()
    finally:
        client.close()

    assert [doc.external_id for doc in docs] == ["doxA1", "doxB2"]  # sheet 被跳过
    assert docs[0].title == "组会纪要 2025-09-10"
    assert docs[0].url.endswith("/wiki/wikA")
    assert docs[0].updated_at.startswith("2025-09-10")


def test_list_documents_from_folder_recurses(tmp_path: Path) -> None:
    client = _feishu(tmp_path, space_id=None, folder_token=FOLDER)
    try:
        docs = client.list_documents()
    finally:
        client.close()

    assert sorted(doc.external_id for doc in docs) == ["fldDoc1", "fldDoc2"]


def test_sync_writes_corpus_loadable_documents_with_comments(tmp_path: Path) -> None:
    client = _feishu(tmp_path)
    try:
        result = sync_source(client, data_dir=tmp_path, target="meeting_minutes")
    finally:
        client.close()

    assert result.listed == 2
    assert len(result.written) == 2
    assert not result.skipped

    corpus = load_corpus(tmp_path)
    doc = corpus.get("meeting_minutes/组会纪要-2025-09-10")
    assert doc is not None
    assert doc.title == "组会纪要 2025-09-10"
    assert doc.kind.value == "meeting_minutes"
    assert doc.meta["date"] == "2025-09-10"
    assert doc.meta["source"] == "feishu"
    assert doc.meta["source_id"] == "doxA1"
    # 正文里的 TODO 能被现有抽取器吃到
    assert any("统一 baseline" in todo.text for todo in corpus.scan_todos())
    # 评论已合并进正文
    assert "先对齐分辨率再比" in doc.text


def test_sync_is_incremental(tmp_path: Path) -> None:
    first = _feishu(tmp_path)
    try:
        sync_source(first, data_dir=tmp_path, target="meeting_minutes")
    finally:
        first.close()

    second = _feishu(tmp_path)
    try:
        result = sync_source(second, data_dir=tmp_path, target="meeting_minutes")
    finally:
        second.close()

    assert result.written == []
    assert len(result.skipped) == 2


def test_sync_dry_run_does_not_write(tmp_path: Path) -> None:
    client = _feishu(tmp_path)
    try:
        result = sync_source(client, data_dir=tmp_path, target="meeting_minutes", dry_run=True)
    finally:
        client.close()

    assert len(result.written) == 2
    assert not (tmp_path / "sources" / "meeting_minutes").exists()
    assert not (tmp_path / "sync_state.json").exists()


def test_sync_rejects_unknown_target(tmp_path: Path) -> None:
    client = _feishu(tmp_path)
    try:
        with pytest.raises(SyncTargetError):
            sync_source(client, data_dir=tmp_path, target="experiment_logs")
    finally:
        client.close()


def test_assign_stems_disambiguates_duplicate_titles() -> None:
    docs = [
        RemoteDoc(external_id="doxAAAA111", title="会议纪要"),
        RemoteDoc(external_id="doxBBBB222", title="会议纪要"),
    ]
    stems = assign_stems(docs)
    assert stems["doxAAAA111"] != stems["doxBBBB222"]
    assert stems["doxAAAA111"] == "会议纪要-doxAAAA1"


def test_auth_failure_raises(tmp_path: Path) -> None:
    client = _feishu(tmp_path, app_secret="wrong-app-secret-00000000")
    try:
        with pytest.raises(FeishuAuthError):
            client.list_documents()
    finally:
        client.close()


def test_missing_app_secret_gives_guidance(tmp_path: Path) -> None:
    manager = SecretsManager(
        name="feishu_app_secret",
        env_var=FEISHU_APP_SECRET_ENV,
        vault=EncryptedVault(tmp_path / "none.enc", passphrase="pw"),
        environ={},
    )
    client = FeishuClient(
        space_id=SPACE_ID,
        app_id=APP_ID,
        settings=Settings(_env_file=None, data_dir=tmp_path, feishu_app_id="", feishu_app_secret=""),
        secrets_manager=manager,
        client=_http(),
    )
    try:
        with pytest.raises(FeishuAuthError) as excinfo:
            client.list_documents()
    finally:
        client.close()
    assert "secrets set" in str(excinfo.value)


def test_credentials_can_come_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"STEPROUTE_FEISHU_APP_ID={APP_ID}\nSTEPROUTE_FEISHU_APP_SECRET={APP_SECRET}\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=env_file, data_dir=tmp_path, http_timeout=5.0)
    client = FeishuClient(space_id=SPACE_ID, settings=settings, client=_http())
    try:
        docs = client.list_documents()
    finally:
        client.close()
    assert [doc.external_id for doc in docs] == ["doxA1", "doxB2"]


def test_settings_repr_hides_app_secret() -> None:
    settings = Settings(feishu_app_id="cli_xxx", feishu_app_secret="super-secret-value-123")
    assert "super-secret-value-123" not in repr(settings)


def test_cli_sync_registered_and_validates_args() -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["sync", "--help"])
    assert help_result.exit_code == 0
    assert "--wiki-space" in help_result.stdout

    bad_source = runner.invoke(app, ["sync", "--source", "dingtalk", "--wiki-space", SPACE_ID])
    assert bad_source.exit_code == 1
    assert "暂不支持来源" in bad_source.stdout

    no_entrance = runner.invoke(app, ["sync"])
    assert no_entrance.exit_code == 1
    assert "请指定数据入口" in no_entrance.stdout