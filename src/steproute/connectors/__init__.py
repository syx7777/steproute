"""外部来源连接器：把远端知识库同步成本地语料。

* :mod:`steproute.connectors.base`   —— 公共模型（``RemoteDoc`` / 协议）；
* :mod:`steproute.connectors.feishu` —— 飞书云文档 / 知识库读取；
* :mod:`steproute.connectors.sync`   —— 落盘式同步（增量水位 + 语料适配）。
"""

from .base import RemoteDoc, RemoteSource
from .feishu import FeishuAuthError, FeishuClient, FeishuError, markdown_to_blocks
from .sync import SyncResult, SyncTargetError, sync_source

__all__ = [
    "FeishuAuthError",
    "FeishuClient",
    "FeishuError",
    "RemoteDoc",
    "RemoteSource",
    "SyncResult",
    "SyncTargetError",
    "markdown_to_blocks",
    "sync_source",
]