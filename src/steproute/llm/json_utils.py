"""稳健的 JSON 解析：从模型自由文本里提取第一个合法 JSON 对象 / 数组。

依次尝试：直接解析 → 去 ``` 代码围栏 → 括号配平扫描（跳过字符串内的括号）。
"""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JSONExtractError(ValueError):
    """无法从文本中解析出 JSON。"""


def extract_json_block(text: str | None) -> Any:
    if text is None:
        raise JSONExtractError("内容为空，无法解析 JSON。")
    candidates: list[str] = [text.strip()]

    fenced = _JSON_FENCE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for candidate in candidates:
        block = scan_balanced(candidate)
        if block is not None:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    snippet = text.strip().replace("\n", " ")[:200]
    raise JSONExtractError(f"无法从输出中解析出 JSON：{snippet!r}")


def scan_balanced(text: str) -> str | None:
    """扫描最外层配平的 ``{...}`` 或 ``[...]``，正确处理字符串与转义。"""
    start = None
    opener = closer = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener, closer = char, ("}" if char == "{" else "]")
            break
    if start is None:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None