"""网络错误诊断：把底层 socket 错误翻译成可执行的提示。

典型场景：配置了代理但代理没启动 → ``WinError 10061 / Connection refused``。
这类错误是**确定性**的，重试没有意义，应该立刻给出可操作的指引。
"""

from __future__ import annotations

from collections.abc import Iterator

_REFUSED_MARKERS = (
    "10061",  # WinError 10061：目标计算机积极拒绝
    "actively refused",
    "connection refused",
    "connectionrefused",
    "econnrefused",
    "connect call failed",
)

_NO_PROXY_TOKENS = frozenset({"", "none", "off", "no", "direct", "false"})


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def is_connection_refused(exc: BaseException) -> bool:
    """判断异常链中是否包含「连接被拒绝」。"""
    for item in _exception_chain(exc):
        text = str(item).lower()
        if any(marker in text for marker in _REFUSED_MARKERS):
            return True
    return False


def parse_proxy_option(value: str | None) -> str | None:
    """把 CLI 传入的 ``--proxy`` 取值规范化为代理 URL 或 ``None``（直连）。

    支持 ``--proxy none`` / ``--proxy off`` / ``--proxy direct`` / ``--proxy ""``
    显式表示直连。
    """
    if value is None:
        return None
    stripped = value.strip()
    return None if stripped.lower() in _NO_PROXY_TOKENS else stripped


def connection_error_message(target: str, proxy: str | None) -> str:
    """生成连接失败时的可执行提示。"""
    if proxy:
        return (
            f"无法连接代理 {proxy}（目标 {target}）：连接被拒绝，通常说明代理未启动或端口不对。\n"
            "  任选一种方式解决：\n"
            "    1) 启动本机代理后重试；\n"
            "    2) 改为直连：设置 STEPROUTE_PROXY_URL=\"\" ，或运行时加 --proxy none；\n"
            "    3) 指定正确端口：STEPROUTE_PROXY_URL=http://127.0.0.1:<port>"
        )
    return (
        f"无法直连 {target}：网络不可达。\n"
        "  若本机需要通过代理访问外网，请设置 STEPROUTE_PROXY_URL=http://127.0.0.1:7892，\n"
        "  或在命令后加 --proxy http://127.0.0.1:7892。"
    )