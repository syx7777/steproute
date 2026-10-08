"""网络代理选项与错误诊断测试。"""

from __future__ import annotations

from steproute.net import (
    connection_error_message,
    is_connection_refused,
    parse_proxy_option,
)


def test_parse_proxy_option_default_none_means_direct() -> None:
    assert parse_proxy_option(None) is None


def test_parse_proxy_option_explicit_direct_tokens() -> None:
    for token in ("", "none", "off", "NONE", " direct ", "false"):
        assert parse_proxy_option(token) is None


def test_parse_proxy_option_keeps_url() -> None:
    assert parse_proxy_option("http://127.0.0.1:7892") == "http://127.0.0.1:7892"


def test_is_connection_refused_detects_winerror() -> None:
    exc = ConnectionRefusedError("[WinError 10061] No connection could be made")
    assert is_connection_refused(exc)


def test_is_connection_refused_detects_nested_cause() -> None:
    inner = OSError("Connection refused")
    outer = RuntimeError("transport failed")
    outer.__cause__ = inner
    assert is_connection_refused(outer)


def test_is_connection_refused_false_for_timeout() -> None:
    assert not is_connection_refused(TimeoutError("timed out"))


def test_connection_error_message_mentions_proxy_fix() -> None:
    message = connection_error_message("https://api.deepseek.com/v1", "http://127.0.0.1:7892")
    assert "7892" in message
    assert "--proxy none" in message


def test_connection_error_message_without_proxy_suggests_proxy() -> None:
    message = connection_error_message("https://api.deepseek.com/v1", None)
    assert "STEPROUTE_PROXY_URL" in message