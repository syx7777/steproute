"""日志与终端输出工具（统一做密钥脱敏）。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .secrets import redact

_CONFIGURED = False


class RedactingFilter(logging.Filter):
    """对日志记录做密钥脱敏，避免密钥随异常/调试信息泄漏。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover
            return True
        scrubbed = redact(message)
        if scrubbed != message:
            record.msg = scrubbed
            record.args = ()
        return True


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """初始化根 logger：彩色控制台 + 可选文件。重复调用安全。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:  # 优先使用 rich 的彩色输出
        from rich.logging import RichHandler

        console: logging.Handler = RichHandler(
            rich_tracebacks=True, show_path=False, markup=False, show_time=False
        )
        console.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    except Exception:  # pragma: no cover
        console = logging.StreamHandler(stream=sys.stderr)
        console.setFormatter(formatter)

    console.addFilter(RedactingFilter())
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RedactingFilter())
        root.addHandler(file_handler)

    # 降低第三方库噪声
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)