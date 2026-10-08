"""LLM 调用的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    """单次调用的 token 计量。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_api(cls, payload: dict[str, Any] | None) -> Usage:
        payload = payload or {}
        completion_details = payload.get("completion_tokens_details") or {}
        prompt_details = payload.get("prompt_tokens_details") or {}
        return cls(
            prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(payload.get("completion_tokens", 0) or 0),
            total_tokens=int(payload.get("total_tokens", 0) or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens", 0) or 0),
            cached_tokens=int(prompt_details.get("cached_tokens", 0) or 0),
        )

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )

    def __sub__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens - other.prompt_tokens,
            completion_tokens=self.completion_tokens - other.completion_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
            reasoning_tokens=self.reasoning_tokens - other.reasoning_tokens,
            cached_tokens=self.cached_tokens - other.cached_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
        }


@dataclass
class ChatResult:
    """一次 LLM 调用的完整结果。"""

    content: str
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: str = ""
    latency_s: float = 0.0
    label: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class TokenLedger:
    """跨调用累计 token 用量，用于成本核算与奖励中的 token 成本项。"""

    def __init__(self) -> None:
        self.calls: int = 0
        self.usage: Usage = Usage()
        self.per_label: dict[str, Usage] = {}

    def record(self, usage: Usage, label: str = "") -> None:
        self.calls += 1
        self.usage = self.usage + usage
        key = label or "default"
        self.per_label[key] = self.per_label.get(key, Usage()) + usage

    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens

    def merge(self, other: TokenLedger) -> None:
        self.calls += other.calls
        self.usage = self.usage + other.usage
        for key, value in other.per_label.items():
            self.per_label[key] = self.per_label.get(key, Usage()) + value

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "usage": self.usage.to_dict(),
            "per_label": {k: v.to_dict() for k, v in self.per_label.items()},
        }

    def reset(self) -> None:
        self.calls = 0
        self.usage = Usage()
        self.per_label.clear()