"""质量评分（rubric）测试。"""

from __future__ import annotations

from steproute.evaluation.rubric import (
    extract_experiment_mentions,
    score_report,
)


def test_reference_is_derived_from_corpus(bundle) -> None:
    reference = bundle.reference
    assert set(reference.experiment_ids) >= {"EXP-001", "EXP-002", "EXP-003", "EXP-004"}
    assert reference.fairness_issue is True
    # 只有"越大越好"的主要指标会被计入落后项
    assert reference.worse_metrics == ["vqa_acc"]
    # 只有可比较性关键项才算不公平来源
    assert reference.mismatched_config_keys == ["frames", "resolution"]


def test_good_report_scores_high(bundle) -> None:
    report = (
        "## 当前研究进展\n"
        "EXP-001 是 baseline，EXP-002 是本方法。方法在 vqa_acc 上落后 2.6 个点。\n"
        "但两者 frames 与 resolution 不一致，属于配置不公平，需要先对齐配置再判断。\n"
        "对齐配置后 EXP-003 达到 74.6，已经超过 baseline。\n"
    )
    tasks = ["统一并对齐配置后重跑公平对比", "补充消融实验验证增益来源"]
    score = score_report(report, tasks, bundle.reference, evidence_count=2)
    assert score.score >= 0.85
    assert score.checks["identifies_fairness_issue"]
    assert score.checks["cites_experiment"]
    assert score.checks["grounded"]


def test_empty_report_scores_low(bundle) -> None:
    score = score_report("", [], bundle.reference, evidence_count=0)
    assert score.score < 0.4
    assert not score.checks["cites_experiment"]
    assert not score.checks["grounded"]


def test_report_without_fairness_mention_loses_points(bundle) -> None:
    report = "## 进展\nEXP-001 与 EXP-002 已跑完，方法准确率较低。\n"
    score = score_report(report, ["继续调参"], bundle.reference, evidence_count=1)
    assert not score.checks["identifies_fairness_issue"]


def test_extract_experiment_mentions(bundle) -> None:
    text = "我们比较了 EXP-001 与 EXP-003，细节见 EXP-004。"
    mentions = extract_experiment_mentions(text, bundle.corpus)
    assert set(mentions) == {"EXP-001", "EXP-003", "EXP-004"}