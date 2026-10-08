"""LLM 后端。

两个实现，共用同一接口 :class:`Backend`：

* :class:`APIBackend` —— 真实调用 OpenAI 兼容网关（DeepSeek 官方 **或** 本地 vLLM）。
  带指数退避重试、连接错误诊断、密钥脱敏、token 计量。
* :class:`SimBackend` —— 纯离线确定性"代理策略"。它读取 prompt 中内嵌的**状态 JSON**，
  用规则生成下一步动作，从而在**不发起任何网络请求**的前提下跑通完整 Agent 闭环，
  用于单元测试与离线演示。
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx

from ..config import ModelSpecConfig, Settings
from ..logging_utils import get_logger
from ..net import connection_error_message, is_connection_refused
from ..secrets import SecretsManager, default_manager
from .schema import ChatResult, Usage

logger = get_logger(__name__)

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
_STATE_BLOCK = re.compile(r"```json\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


class BackendError(RuntimeError):
    """后端调用错误基类。"""


class BackendTransientError(BackendError):
    """可重试的瞬时错误。"""


class Backend(Protocol):
    """LLM 后端接口。"""

    name: str

    def complete(
        self,
        spec: ModelSpecConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        label: str = "",
    ) -> ChatResult: ...


# --------------------------------------------------------------------------- #
# 真实 API 后端
# --------------------------------------------------------------------------- #

class APIBackend:
    """OpenAI 兼容 Chat Completions 后端（每个 endpoint 复用一个 httpx.Client）。"""

    name = "api"

    def __init__(self, settings: Settings, *, api_key: str | None = None, secrets_manager: SecretsManager | None = None) -> None:
        self.settings = settings
        self._secrets = secrets_manager or default_manager()
        self._api_key = api_key
        self._clients: dict[str, httpx.Client] = {}

    # -- 资源 -------------------------------------------------------------- #
    @property
    def api_key(self) -> str:
        if self._api_key is None:
            self._api_key = self._secrets.get_llm_api_key()
        return self._api_key

    def _client_for(self, endpoint: str) -> httpx.Client:
        if endpoint not in self._clients:
            self._clients[endpoint] = httpx.Client(
                base_url=endpoint.rstrip("/"),
                timeout=httpx.Timeout(self.settings.llm_timeout, connect=20.0),
                proxy=self.settings.proxy_for_httpx(),
                trust_env=self.settings.trust_env_proxy,
                follow_redirects=True,
            )
        return self._clients[endpoint]

    def close(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    # -- 调用 -------------------------------------------------------------- #
    def complete(
        self,
        spec: ModelSpecConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        label: str = "",
    ) -> ChatResult:
        endpoint = spec.endpoint or self.settings.llm_base_url
        body: dict[str, Any] = {
            "model": spec.model,
            "messages": [dict(m) for m in messages],
            "temperature": spec.temperature if temperature is None else temperature,
            "max_tokens": spec.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        effort = reasoning_effort or spec.reasoning_effort
        if effort:
            body["reasoning_effort"] = effort
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        payload = self._post_with_retry(endpoint, body, label=label)
        latency = time.perf_counter() - started
        return self._to_result(payload, spec=spec, latency=latency, label=label)

    def _post_with_retry(self, endpoint: str, body: dict[str, Any], *, label: str) -> dict[str, Any]:
        attempts = max(1, self.settings.llm_max_retries)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                client = self._client_for(endpoint)
                response = client.post(
                    "/chat/completions",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if is_connection_refused(exc):
                    # 代理未启动 / 端口不对属于确定性失败，重试无意义
                    raise BackendError(
                        connection_error_message(endpoint, self.settings.proxy_for_httpx())
                    ) from exc
                last_error = BackendTransientError(f"网络异常: {exc}")
                logger.warning("[%s] LLM 网络异常（第 %d/%d 次）：%s", label, attempt, attempts, exc)
                self._backoff(attempt)
                continue

            if response.status_code in _RETRYABLE_STATUS:
                last_error = BackendTransientError(f"HTTP {response.status_code}: {response.text[:200]}")
                logger.warning("[%s] LLM 可重试状态码 %d（第 %d/%d 次）", label, response.status_code, attempt, attempts)
                self._backoff(attempt)
                continue
            if response.status_code >= 400:
                raise BackendError(f"LLM 调用失败 HTTP {response.status_code}: {response.text[:500]}")

            try:
                return response.json()
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise BackendError(f"LLM 返回非 JSON 响应: {response.text[:300]}") from exc

        raise last_error or BackendError("LLM 调用失败：重试次数耗尽")

    @staticmethod
    def _backoff(attempt: int) -> None:
        time.sleep(min(2.0 ** (attempt - 1), 8.0))

    @staticmethod
    def _to_result(payload: dict[str, Any], *, spec: ModelSpecConfig, latency: float, label: str) -> ChatResult:
        choices = payload.get("choices") or []
        choice = next(iter(choices), {}) if isinstance(choices, list) else {}
        message = choice.get("message") or {}
        return ChatResult(
            content=(message.get("content") or "").strip(),
            reasoning=(message.get("reasoning_content") or "").strip(),
            usage=Usage.from_api(payload.get("usage")),
            model=payload.get("model", spec.model),
            finish_reason=choice.get("finish_reason", "") or "",
            latency_s=latency,
            label=label,
        )


# --------------------------------------------------------------------------- #
# 离线仿真后端
# --------------------------------------------------------------------------- #

class SimBackend:
    """离线确定性后端：用规则代理策略跑通 Agent 闭环（不联网）。

    它从 prompt 最后一个 user 消息里解析 `````json`` 包裹的状态块，
    依据"已完成工具集合"推进一条固定但通用的参考轨迹：

        search_corpus → read_document → list_experiments
        → compare_experiments → extract_todos → finalize

    这样可以零成本验证：状态构造、路由、调度、工具执行、报告生成全链路。
    """

    name = "sim"

    def complete(
        self,
        spec: ModelSpecConfig,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
        label: str = "",
    ) -> ChatResult:
        state = _extract_state(messages)
        action = _reference_action(state)
        content = json.dumps(action, ensure_ascii=False)
        # 伪造一次"看起来合理"的用量，供成本 / token 统计走通
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        prompt_tokens = max(32, prompt_chars // 4)
        completion_tokens = max(16, len(content) // 4)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            # 仿真里假设首轮之后有一定前缀命中率
            cached_tokens=int(prompt_tokens * (0.6 if state.get("step_index", 0) else 0.0)),
        )
        return ChatResult(
            content=content,
            reasoning="[sim] 依据状态按参考策略选择下一步动作。",
            usage=usage,
            model=f"{spec.model}@sim",
            finish_reason="stop",
            latency_s=simulated_latency(spec, prompt_tokens, completion_tokens),
            label=label,
        )


def simulated_latency(spec: ModelSpecConfig, prompt_tokens: int, completion_tokens: int) -> float:
    """按模型画像合成一次调用的延迟，让离线运行的延迟统计非零且可比较。"""
    return (
        spec.base_latency_s
        + prompt_tokens / 1000.0 * spec.latency_per_1k_prompt_s
        + completion_tokens / 1000.0 * spec.latency_per_1k_completion_s
    )


def _extract_state(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """从消息中解析最后一块 ```json 状态块。"""
    for message in reversed(list(messages)):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for match in reversed(list(_STATE_BLOCK.finditer(content))):
            try:
                parsed = json.loads(match.group("body"))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "step_index" in parsed:
                return parsed
    return {}


def _reference_action(state: dict[str, Any]) -> dict[str, Any]:
    """确定性的参考动作策略（供离线仿真使用）。

    按固定顺序推进一条通用轨迹；每一步都带着 ``next_step_type``，
    以便 Harness 预测下一轮难度（与真实模型的行为契约一致）。
    """
    done = set(state.get("completed_tools") or [])
    task = str(state.get("task") or "整理科研进展")
    retrieved = [str(x) for x in (state.get("retrieved_ids") or [])]
    experiment_ids = [str(x) for x in (state.get("experiment_ids") or [])]
    evidence = list(state.get("evidence") or [])
    todos = [str(t) for t in (state.get("todos") or [])]
    flags = state.get("flags") or {}

    if "search_corpus" not in done:
        return _action("retrieve", "先在语料中检索与任务相关的周报、纪要与实验记录。", "read",
                       "search_corpus", {"query": task, "top_k": 5})

    if "read_document" not in done and retrieved:
        return _action("read", "阅读检索到的最高相关文档，提取关键结论与实验编号。", "extract",
                       "read_document", {"doc_id": retrieved[0]})

    if "list_experiments" not in done:
        return _action("extract", "列出全部实验记录，拿到可比对的实验编号与指标。", "compare",
                       "list_experiments", {})

    if "compare_experiments" not in done and len(experiment_ids) >= 2:
        return _action("compare", "对方法与 baseline 做一次比较，并检查配置是否公平。", "analyze",
                       "compare_experiments", {"experiment_a": experiment_ids[0], "experiment_b": experiment_ids[1]})

    if "record_finding" not in done and not evidence:
        conflict = bool(flags.get("numeric_conflict"))
        claim = (
            "对比发现：方法在指标上落后于 baseline，且两者实验配置不一致，当前对比并不公平，需先对齐配置。"
            if conflict
            else "对比发现：方法与 baseline 的配置基本一致，需从方法本身排查未超过 baseline 的原因。"
        )
        return _action("analyze", "把比较得到的结论登记为可引用证据。", "synthesize",
                       "record_finding",
                       {"claim": claim, "source_ids": (experiment_ids[:2] or retrieved[:2]), "kind": "analysis"})

    if "extract_todos" not in done:
        return _action("extract", "从周报评论与会议纪要中抽取尚未完成的 TODO。", "synthesize",
                       "extract_todos", {})

    report_lines = ["## 当前研究进展", ""]
    report_lines += [f"- {str(e.get('claim', ''))}" for e in evidence] or ["- 暂未收集到明确结论。"]
    report_lines += ["", "## 下一步任务", ""]
    next_tasks = todos[:6] or ["统一评测配置后重新对比 baseline"]
    report_lines += [f"{i + 1}. {t}" for i, t in enumerate(next_tasks)]
    report_lines += ["", "## 依据来源", ""]
    report_lines += [f"- {doc_id}" for doc_id in (retrieved[:3] or experiment_ids[:3])]
    return _action("final", "证据与 TODO 已齐备，输出进度报告与下一步任务。", "final",
                   "finalize", {"report": "\n".join(report_lines), "next_tasks": next_tasks})


def _action(step_type: str, thought: str, next_step_type: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_type": step_type,
        "thought": thought,
        "next_step_type": next_step_type,
        "tool": tool,
        "args": args,
    }