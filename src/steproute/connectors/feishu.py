"""飞书（Feishu / Lark）连接器：把云文档与知识库读取为规范化的 :class:`RemoteDoc`。

鉴权
----
自建应用用 ``App ID`` + ``App Secret`` 换取 ``tenant_access_token``（有效期 2 小时，本地缓存）。
``App ID`` 走运行期配置（``STEPROUTE_FEISHU_APP_ID``），``App Secret`` 走
:mod:`steproute.secrets`（环境变量 ``STEPROUTE_FEISHU_APP_SECRET`` → 系统凭据管理器 →
加密密钥库），**不会明文落盘**。

用到的接口
----------
* ``POST /auth/v3/tenant_access_token/internal``  —— 换取租户凭证；
* ``GET  /wiki/v2/spaces/:space_id/nodes``        —— 知识库节点（递归子节点）；
* ``GET  /drive/v1/files``                        —— 云空间文件夹清单（递归子文件夹）；
* ``GET  /docx/v1/documents/:document_id/raw_content`` —— 文档正文（纯文本）；
* ``GET  /drive/v1/files/:file_token/comments``   —— 文档评论。

写盘由 :mod:`steproute.connectors.sync` 负责，本模块只负责"取数"。
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import httpx

from ..config import Settings
from ..logging_utils import get_logger
from ..net import connection_error_message, is_connection_refused
from ..secrets import KNOWN_SECRET_ENV, SecretsManager, manager_for, register_secret
from .base import RemoteDoc, RemoteError

logger = get_logger(__name__)

FEISHU_APP_SECRET_NAME = "feishu_app_secret"
FEISHU_APP_SECRET_ENV = KNOWN_SECRET_ENV[FEISHU_APP_SECRET_NAME]
DEFAULT_BASE_URL = "https://open.feishu.cn/open-apis"
DEFAULT_SITE_URL = "https://feishu.cn"

_PAGE_SIZE_WIKI = 50
_PAGE_SIZE_DRIVE = 200
_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
#: 业务码：凭证 / 权限类错误（直接给出可执行提示，不重试）。
_AUTH_CODES = {10003, 10014, 99991661, 99991663, 99991664, 99991668, 99991672, 99991679}
#: 业务码：触发频率限制。
_RATE_LIMIT_CODE = 99991400


class FeishuError(RemoteError):
    """飞书同步错误基类。"""


class FeishuAuthError(FeishuError):
    """鉴权 / 权限失败：凭证缺失、无效或应用无该资源权限。"""


class FeishuClient:
    """飞书开放平台客户端（自建应用，tenant_access_token）。

    绑定一个数据入口：知识库 ``space_id`` **或** 云空间文件夹 ``folder_token``
    （``folder_token=""`` 表示个人云空间根目录）。
    """

    name = "feishu"

    def __init__(
        self,
        *,
        space_id: str | None = None,
        folder_token: str | None = None,
        app_id: str | None = None,
        app_secret: str | None = None,
        base_url: str | None = None,
        site_url: str | None = None,
        settings: Settings | None = None,
        secrets_manager: SecretsManager | None = None,
        client: httpx.Client | None = None,
        include_comments: bool = True,
    ) -> None:
        self.settings = settings or Settings()
        self.space_id = space_id
        self.folder_token = folder_token
        self.base_url = (base_url or self.settings.feishu_base_url).rstrip("/")
        self.site_url = (site_url or self.settings.feishu_site_url).rstrip("/")
        self.include_comments = include_comments
        self._app_id = app_id
        self._app_secret = app_secret
        self._secrets = secrets_manager or manager_for(FEISHU_APP_SECRET_NAME)
        self._client = client
        self._owns_client = client is None
        self._tenant_token: str | None = None
        self._tenant_expiry: float = 0.0

    # -- 资源 -------------------------------------------------------------- #
    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.settings.http_timeout, connect=15.0),
                proxy=self.settings.proxy_for_httpx(),
                trust_env=self.settings.trust_env_proxy,
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    # -- 凭证 -------------------------------------------------------------- #
    @property
    def app_id(self) -> str:
        value = (self._app_id or self.settings.feishu_app_id or "").strip()
        if not value:
            raise FeishuAuthError(
                "未配置飞书应用 App ID。请任选一种方式：\n"
                "  1) 设置环境变量 STEPROUTE_FEISHU_APP_ID=cli_xxx\n"
                "  2) 运行时加 --app-id cli_xxx"
            )
        return value

    @property
    def app_secret(self) -> str:
        if self._app_secret is not None:
            return self._app_secret
        # 便捷路径：显式配置（.env / STEPROUTE_FEISHU_APP_SECRET）优先。
        configured = (self.settings.feishu_app_secret or "").strip()
        if configured:
            register_secret(configured)
            self._app_secret = configured
            return configured
        # 安全路径：环境变量 → 系统凭据管理器 → 加密密钥库。
        resolved = self._secrets.resolve()
        if resolved is None:
            raise FeishuAuthError(
                "未找到飞书应用 App Secret。请任选一种方式配置：\n"
                f"  1) 推荐：uv run steproute secrets set --name {FEISHU_APP_SECRET_NAME}\n"
                f"  2) 写进 .env：STEPROUTE_FEISHU_APP_SECRET=<secret>（明文落盘）\n"
                f"  3) 临时：$env:{FEISHU_APP_SECRET_ENV} = '<secret>'"
            )
        self._app_secret = resolved.value
        return resolved.value

    def tenant_token(self) -> str:
        if self._tenant_token and time.time() < self._tenant_expiry - 60:
            return self._tenant_token
        payload = self._request(
            "POST",
            "/auth/v3/tenant_access_token/internal",
            body={"app_id": self.app_id, "app_secret": self.app_secret},
            authenticated=False,
        )
        token = str(payload.get("tenant_access_token") or "")
        if not token:
            raise FeishuAuthError(f"飞书未返回 tenant_access_token：{payload.get('msg', '')}")
        self._tenant_token = token
        self._tenant_expiry = time.time() + float(payload.get("expire") or 7200)
        return token

    # -- 请求 -------------------------------------------------------------- #
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.tenant_token()}"

        last_error: FeishuError | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = self._http().request(method, path, params=params, json=body, headers=headers)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if is_connection_refused(exc):
                    raise FeishuError(
                        connection_error_message(self.base_url, self.settings.proxy_for_httpx())
                    ) from exc
                last_error = FeishuError(f"飞书网络异常: {exc}")
                logger.warning("飞书网络异常（第 %d/%d 次）：%s", attempt, _MAX_ATTEMPTS, exc)
                self._backoff(attempt)
                continue

            if response.status_code in (401, 403):
                raise FeishuAuthError(
                    f"飞书鉴权失败 HTTP {response.status_code}：请确认 App ID/Secret 正确、"
                    "且应用已被添加为目标知识库/文档的协作者。"
                )
            if response.status_code in _RETRYABLE_STATUS:
                last_error = FeishuError(f"飞书可重试状态码 HTTP {response.status_code}")
                logger.warning("飞书可重试状态码 %d（第 %d/%d 次）", response.status_code, attempt, _MAX_ATTEMPTS)
                self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise FeishuError(f"飞书请求失败 HTTP {response.status_code}: {response.text[:300]}")

            payload = response.json()
            code = payload.get("code")
            if code in (0, None):
                return payload
            if code == _RATE_LIMIT_CODE:
                last_error = FeishuError("飞书接口触发频率限制")
                self._backoff(attempt)
                continue
            if code in _AUTH_CODES:
                raise FeishuAuthError(f"飞书鉴权/权限失败 code={code}: {payload.get('msg', '')}")
            raise FeishuError(f"飞书接口返回错误 code={code}: {payload.get('msg', '')}")

        raise last_error or FeishuError("飞书请求失败：重试次数耗尽")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2.0 ** (attempt - 1), 8.0))

    # -- 解析 -------------------------------------------------------------- #
    @staticmethod
    def _ts_to_iso(value: Any) -> str:
        """飞书时间戳（秒，字符串）→ ISO 8601。"""
        if value is None or value == "":
            return ""
        try:
            stamp = int(str(value).strip())
        except (TypeError, ValueError):
            return str(value)
        return datetime.fromtimestamp(stamp, UTC).isoformat(timespec="seconds")

    def _wiki_doc(self, node: dict[str, Any], space_id: str) -> RemoteDoc:
        token = str(node.get("obj_token") or "")
        node_token = str(node.get("node_token") or "")
        return RemoteDoc(
            external_id=token,
            title=str(node.get("title") or token),
            container=space_id,
            url=f"{self.site_url}/wiki/{node_token}" if node_token else "",
            updated_at=self._ts_to_iso(node.get("obj_edit_time")),
            obj_type=str(node.get("obj_type") or ""),
        )

    def _folder_doc(self, item: dict[str, Any], container: str) -> RemoteDoc:
        token = str(item.get("token") or "")
        return RemoteDoc(
            external_id=token,
            title=str(item.get("name") or token),
            container=container,
            url=str(item.get("url") or ""),
            updated_at=self._ts_to_iso(item.get("modified_time")),
            obj_type=str(item.get("type") or ""),
        )

    # -- 列出 -------------------------------------------------------------- #
    def list_documents(self) -> list[RemoteDoc]:
        """按绑定的入口列出可同步的文档（仅 ``docx``，不含子类型说明）。"""
        if self.space_id is not None:
            docs = self._list_wiki_nodes(self.space_id)
        elif self.folder_token is not None:
            docs = self._list_folder_docs(self.folder_token)
        else:
            raise FeishuError("未指定数据入口：请传 --wiki-space <space_id> 或 --folder <folder_token>。")

        unique: dict[str, RemoteDoc] = {}
        for doc in docs:
            if doc.external_id and doc.external_id not in unique:
                unique[doc.external_id] = doc
        logger.info("飞书文档列表：%d 篇（去重后 %d 篇）", len(docs), len(unique))
        return list(unique.values())

    def _iter_wiki_nodes(self, space_id: str, parent_node_token: str | None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": _PAGE_SIZE_WIKI}
            if parent_node_token:
                params["parent_node_token"] = parent_node_token
            if page_token:
                params["page_token"] = page_token
            payload = self._request("GET", f"/wiki/v2/spaces/{space_id}/nodes", params=params)
            data = payload.get("data") or {}
            items.extend(item for item in (data.get("items") or []) if isinstance(item, dict))
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
            if not page_token:
                break
        return items

    def _list_wiki_nodes(self, space_id: str) -> list[RemoteDoc]:
        docs: list[RemoteDoc] = []
        pending: list[str | None] = [None]
        visited: set[str] = set()
        while pending:
            parent = pending.pop(0)
            for node in self._iter_wiki_nodes(space_id, parent):
                node_token = str(node.get("node_token") or "")
                if not node_token or node_token in visited:
                    continue
                visited.add(node_token)
                if node.get("has_child"):
                    pending.append(node_token)
                if str(node.get("obj_type")) == "docx":
                    docs.append(self._wiki_doc(node, space_id))
        return docs

    def _iter_folder_items(self, folder_token: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": _PAGE_SIZE_DRIVE}
            if folder_token:
                params["folder_token"] = folder_token
            if page_token:
                params["page_token"] = page_token
            payload = self._request("GET", "/drive/v1/files", params=params)
            data = payload.get("data") or {}
            items.extend(item for item in (data.get("files") or []) if isinstance(item, dict))
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
            if not page_token:
                break
        return items

    def _list_folder_docs(self, root: str) -> list[RemoteDoc]:
        docs: list[RemoteDoc] = []
        pending: list[str] = [root]
        visited: set[str] = set()
        while pending:
            folder = pending.pop(0)
            if folder in visited:
                continue
            visited.add(folder)
            for item in self._iter_folder_items(folder):
                item_type = str(item.get("type") or "")
                if item_type == "folder":
                    token = str(item.get("token") or "")
                    if token:
                        pending.append(token)
                elif item_type == "docx":
                    docs.append(self._folder_doc(item, folder or "root"))
        return docs

    # -- 取数 -------------------------------------------------------------- #
    def _fetch_body(self, document_id: str) -> str:
        payload = self._request("GET", f"/docx/v1/documents/{document_id}/raw_content")
        return str((payload.get("data") or {}).get("content") or "").strip()

    def _fetch_comments(self, file_token: str) -> list[str]:
        comments: list[str] = []
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {"file_type": "docx", "page_size": 100}
            if page_token:
                params["page_token"] = page_token
            payload = self._request("GET", f"/drive/v1/files/{file_token}/comments", params=params)
            data = payload.get("data") or {}
            for item in data.get("items") or []:
                text = self._render_comment(item)
                if text:
                    comments.append(text)
            if not data.get("has_more"):
                break
            page_token = data.get("page_token")
            if not page_token:
                break
        return comments

    @staticmethod
    def _render_comment(item: dict[str, Any]) -> str:
        replies = ((item.get("reply_list") or {}).get("replies")) or []
        parts: list[str] = []
        for reply in replies:
            elements = ((reply.get("content") or {}).get("elements")) or []
            texts = [
                str((element.get("text_run") or {}).get("text") or "")
                for element in elements
                if isinstance(element, dict)
            ]
            joined = "".join(texts).strip()
            if joined:
                parts.append(joined)
        return " / ".join(parts)

    def fetch_document(self, doc: RemoteDoc) -> RemoteDoc:
        """补齐正文与评论。"""
        body = self._fetch_body(doc.external_id)
        comments: tuple[str, ...] = ()
        if self.include_comments:
            comments = tuple(self._fetch_comments(doc.external_id))
        return replace(doc, body=body, comments=comments)

    # -- 写入（周报落地用）-------------------------------------------------- #
    def iter_space_nodes(self, space_id: str, *, recursive: bool = True) -> list[dict[str, Any]]:
        """列出知识空间内全部节点（``recursive=True`` 时递归子节点）。"""
        nodes: list[dict[str, Any]] = []
        pending: list[str | None] = [None]
        visited: set[str] = set()
        while pending:
            parent = pending.pop(0)
            for node in self._iter_wiki_nodes(space_id, parent):
                node_token = str(node.get("node_token") or "")
                if not node_token or node_token in visited:
                    continue
                visited.add(node_token)
                nodes.append(node)
                if recursive and node.get("has_child"):
                    pending.append(node_token)
        return nodes

    def find_template_token(self, space_id: str, keyword: str) -> str:
        """在知识空间里找名称含 ``keyword`` 的 ``docx`` 文档，返回其 obj_token。"""
        if not keyword:
            return ""
        for node in self.iter_space_nodes(space_id):
            title = str(node.get("title") or "")
            if keyword in title and str(node.get("obj_type")) == "docx":
                token = str(node.get("obj_token") or "")
                if token:
                    logger.info("命中模板文档：%s（%s）", title, token)
                    return token
        return ""

    def read_document_text(self, document_id: str) -> str:
        """读取文档纯文本（用于提取模板结构）。"""
        return self._fetch_body(document_id)

    def create_space_node(
        self,
        space_id: str,
        *,
        title: str,
        parent_node_token: str | None = None,
        obj_type: str = "docx",
    ) -> dict[str, Any]:
        """在知识空间下新建一个 ``docx`` 节点，返回节点信息。"""
        body: dict[str, Any] = {"obj_type": obj_type, "node_type": "origin", "title": title}
        if parent_node_token:
            body["parent_node_token"] = parent_node_token
        payload = self._request("POST", f"/wiki/v2/spaces/{space_id}/nodes", body=body)
        return (payload.get("data") or {}).get("node") or {}

    def append_markdown(self, document_id: str, markdown: str) -> int:
        """把 Markdown 转成文档块并追加到文档末尾，返回写入的块数。"""
        blocks = markdown_to_blocks(markdown)
        if not blocks:
            return 0
        created = 0
        for index in range(0, len(blocks), _BLOCK_BATCH):
            batch = blocks[index : index + _BLOCK_BATCH]
            body = {"index": created, "children": batch}
            self._request(
                "POST",
                f"/docx/v1/documents/{document_id}/blocks/{document_id}/children",
                body=body,
            )
            created += len(batch)
        logger.info("已写入文档 %s：%d 个块", document_id, created)
        return created


# --------------------------------------------------------------------------- #
# Markdown → 文档块
# --------------------------------------------------------------------------- #

#: 单次「创建块」请求最多携带的子块数（飞书上限 50，留余量）。
_BLOCK_BATCH = 40

_HEADING_TYPES = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11}
_HEADING_FIELDS = {
    3: "heading1",
    4: "heading2",
    5: "heading3",
    6: "heading4",
    7: "heading5",
    8: "heading6",
    9: "heading7",
    10: "heading8",
    11: "heading9",
}
_FENCE_RE = re.compile(r"^\s*```\s*(?P<lang>.*)$")
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,9})\s+(?P<body>.+)$")
_TODO_RE = re.compile(r"^\s*[-*+]\s*\[(?P<mark>[ xX])\]\s*(?P<body>.+)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(?P<body>.+)$")
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+(?P<body>.+)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(?P<body>.*)$")
_DIVIDER_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def _elements(content: str) -> list[dict[str, Any]]:
    return [{"text_run": {"content": content}}]


def _simple_block(block_type: int, field: str, content: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"elements": _elements(content)}
    payload.update(extra)
    return {"block_type": block_type, field: payload}


def markdown_to_blocks(markdown: str) -> list[dict[str, Any]]:
    """把 Markdown 转成飞书新版文档的块数组（支持标题/列表/待办/引用/代码/分割线）。"""
    blocks: list[dict[str, Any]] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []

    def flush_code() -> None:
        nonlocal code_lines, code_lang
        if code_lines:
            blocks.append(
                {
                    "block_type": 14,
                    "code": {"elements": _elements("\n".join(code_lines)), "style": {"language": code_lang or "plaintext"}},
                }
            )
        code_lines = []
        code_lang = ""

    for raw in markdown.splitlines():
        fence = _FENCE_RE.match(raw)
        if fence:
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
                code_lang = (fence.group("lang") or "").strip()
            continue
        if in_code:
            code_lines.append(raw)
            continue
        if not raw.strip():
            continue
        if _DIVIDER_RE.match(raw):
            blocks.append({"block_type": 22, "divider": {}})
            continue

        heading = _HEADING_RE.match(raw.strip())
        if heading:
            block_type = _HEADING_TYPES[len(heading.group("hashes"))]
            blocks.append(_simple_block(block_type, _HEADING_FIELDS[block_type], heading.group("body").strip()))
            continue

        todo = _TODO_RE.match(raw)
        if todo:
            blocks.append(
                _simple_block(
                    17,
                    "todo",
                    todo.group("body").strip(),
                    style={"done": todo.group("mark").lower() == "x"},
                )
            )
            continue

        bullet = _BULLET_RE.match(raw)
        if bullet:
            blocks.append(_simple_block(12, "bullet", bullet.group("body").strip()))
            continue

        ordered = _ORDERED_RE.match(raw)
        if ordered:
            blocks.append(_simple_block(13, "ordered", ordered.group("body").strip()))
            continue

        quote = _QUOTE_RE.match(raw)
        if quote:
            blocks.append(_simple_block(15, "quote", quote.group("body").strip()))
            continue

        blocks.append(_simple_block(2, "text", raw.strip()))

    if in_code:
        flush_code()
    return blocks