"""质量评分（rubric）：对真实运行产出的进度报告与下一步任务做可复现打分。

评估基准不是硬编码的，而是**从语料里推导**出来的「应有事实」：

* 出现了哪些实验编号；
* 是否存在"配置不一致导致对比不公平"的情况（读出 config 差异与指标差异）。

这样换一批材料也能自动适配，避免"为样例写死答案"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..corpus.model import (
    COMPARABILITY_KEYS,
    Corpus,
    experiment_role,
    metric_direction,
)
from ..corpus.retriever import tokenize

_FAIRNESS_PATTERNS = (
    "公平", "对齐", "配置不一致", "不一致", "统一配置", "分辨率", "resolution", "帧数", "frames", "预处理",
)
_UNDERPERFORM_PATTERNS = ("未超过", "没有超过", "落后", "低于", "不如", "差于", "baseline", "基线")
_ACTION_VERBS = ("对齐", "统一", "补充", "补齐", "复现", "验证", "重跑", "重做", "消融", "对比", "排查", "增加")
_PROGRESS_PATTERNS = ("进展", "进度", "已完成", "现状", "梳理")


@dataclass
class Reference:
    """从语料推导出的"应有事实"。"""

    experiment_ids: list[str] = field(default_factory=list)
    fairness_issue: bool = False
    worse_metrics: list[str] = field(default_factory=list)
    mismatched_config_keys: list[str] = field(default_factory=list)
    todo_keywords: list[str] = field(default_factory=list)


@dataclass
class RubricScore:
    """质量评分结果。"""

    score: float
    checks: dict[str, bool] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 4), "checks": self.checks, "detail": self.detail}


# --------------------------------------------------------------------------- #
# 参考事实
# --------------------------------------------------------------------------- #

def build_reference(corpus: Corpus) -> Reference:
    """从语料推导评估基准（方向感知：只有"方法在越大越好的指标上更差"才算落后）。"""
    reference = Reference(experiment_ids=list(corpus.experiment_ids))

    experiments = [corpus.experiment(exp_id) for exp_id in corpus.experiment_ids]
    experiments = [doc for doc in experiments if doc is not None]
    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            a, b = experiments[i], experiments[j]
            if experiment_role(a) == "baseline" and experiment_role(b) != "baseline":
                base, method = a, b
            elif experiment_role(b) == "baseline" and experiment_role(a) != "baseline":
                base, method = b, a
            else:
                base, method = a, b

            cfg_a: dict[str, Any] = a.meta.get("config") or {}
            cfg_b: dict[str, Any] = b.meta.get("config") or {}
            mismatches = sorted(
                key for key in set(cfg_a) | set(cfg_b) if key in COMPARABILITY_KEYS and cfg_a.get(key) != cfg_b.get(key)
            )

            base_metrics: dict[str, Any] = base.meta.get("metrics") or {}
            method_metrics: dict[str, Any] = method.meta.get("metrics") or {}
            worse: list[str] = []
            for metric in set(method_metrics) & set(base_metrics):
                if metric_direction(metric) <= 0:
                    continue  # 只以"越大越好"的主要指标判断优劣
                try:
                    if float(method_metrics[metric]) < float(base_metrics[metric]):
                        worse.append(metric)
                except (TypeError, ValueError):
                    continue

            if mismatches and worse:
                reference.fairness_issue = True
                reference.mismatched_config_keys = sorted(set(reference.mismatched_config_keys) | set(mismatches))
                reference.worse_metrics = sorted(set(reference.worse_metrics) | set(worse))

    reference.todo_keywords = [item.text for item in corpus.scan_todos() if item.status == "open"][:8]
    return reference


# --------------------------------------------------------------------------- #
# 打分
# --------------------------------------------------------------------------- #

def score_report(
    report: str,
    next_tasks: list[str],
    reference: Reference,
    *,
    evidence_count: int = 0,
) -> RubricScore:
    """对报告与下一步任务打分（0~1）。"""
    text = report or ""
    tasks_text = "\n".join(next_tasks or [])
    combined = f"{text}\n{tasks_text}"
    lowered = combined.lower()

    checks: dict[str, bool] = {}
    checks["has_progress_section"] = any(pattern in combined for pattern in _PROGRESS_PATTERNS)
    checks["mentions_underperform"] = any(pattern in lowered for pattern in _UNDERPERFORM_PATTERNS)
    checks["cites_experiment"] = any(exp_id and exp_id.lower() in lowered for exp_id in reference.experiment_ids)
    checks["has_multiple_tasks"] = len(next_tasks or []) >= 2
    checks["tasks_actionable"] = any(any(verb in task for verb in _ACTION_VERBS) for task in (next_tasks or []))
    checks["grounded"] = evidence_count > 0

    if reference.fairness_issue:
        checks["identifies_fairness_issue"] = any(pattern in lowered for pattern in _FAIRNESS_PATTERNS)
    else:
        checks["identifies_fairness_issue"] = True  # 无公平性问题时不扣分

    # 语义覆盖率：报告与"应有结论"关键词的重合度
    expected_tokens: set[str] = set()
    for keyword in [*_FAIRNESS_PATTERNS, *_UNDERPERFORM_PATTERNS, *reference.mismatched_config_keys, *reference.worse_metrics]:
        expected_tokens.update(tokenize(str(keyword)))
    report_tokens = set(tokenize(combined))
    coverage = (len(expected_tokens & report_tokens) / len(expected_tokens)) if expected_tokens else 0.0
    checks["semantic_coverage"] = coverage >= 0.35

    weights = {
        "has_progress_section": 0.10,
        "mentions_underperform": 0.15,
        "cites_experiment": 0.15,
        "has_multiple_tasks": 0.10,
        "tasks_actionable": 0.15,
        "grounded": 0.10,
        "identifies_fairness_issue": 0.15,
        "semantic_coverage": 0.10,
    }
    total_weight = sum(weights.values())
    score = sum(weights[key] for key, passed in checks.items() if passed and key in weights) / total_weight

    return RubricScore(
        score=score,
        checks=checks,
        detail={
            "coverage": round(coverage, 3),
            "fairness_issue_expected": reference.fairness_issue,
            "mismatched_config_keys": reference.mismatched_config_keys,
            "worse_metrics": reference.worse_metrics,
        },
    )


def extract_experiment_mentions(text: str, corpus: Corpus) -> list[str]:
    """报告里被明确提到的实验编号。"""
    mentions: list[str] = []
    for exp_id in corpus.experiment_ids:
        if exp_id and re.search(re.escape(exp_id), text, re.IGNORECASE):
            mentions.append(exp_id)
    return mentions