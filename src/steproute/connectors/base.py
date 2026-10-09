"""外部来源连接器的公共模型与协议（与具体平台解耦）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class RemoteError(RuntimeError):
    """外部来源错误基类。"""


@dataclass(frozen=True)
class RemoteDoc:
    """远端文档的规范化表示。

    ``list_documents`` 阶段只填充元数据，``body`` / ``comments`` 为空；
    ``fetch_document`` 阶段补齐正文与评论。
    """

    #: 用于取正文的唯一标识（飞书里是 docx 的 document_id / obj_token）。
    external_id: str
    title: str
    #: 所属容器（知识库 space_id / 文件夹 token）。
    container: str = ""
    url: str = ""
    updated_at: str = ""
    author: str = ""
    obj_type: str = ""
    body: str = ""
    comments: tuple[str, ...] = field(default_factory=tuple)

    @property
    def doc_key(self) -> str:
        """状态水位与去重使用的稳定键。"""
        return self.external_id or self.title


class RemoteSource(Protocol):
    """可同步的外部来源（飞书 / 钉钉 …）。"""

    name: str

    def list_documents(self) -> list[RemoteDoc]: ...

    def fetch_document(self, doc: RemoteDoc) -> RemoteDoc: ...

    def close(self) -> None: ...