from __future__ import annotations

from agentic_rag_rl.evaluation import exact_match, hop_recall, over_extension, premature_collapse, step_alignment, token_f1


def test_exact_match_and_f1() -> None:
    assert exact_match("52.33亿元", "52.33 亿元") == 1.0
    assert token_f1("红旗连锁", "红旗连锁") == 1.0
    assert token_f1("红旗连锁", "永辉超市") == 0.0


def test_hop_aware_metrics() -> None:
    predicted = ["hq_0001", "hq_0002"]
    gold = ["hq_0001", "hq_0002"]

    assert hop_recall(predicted, gold) == 1.0
    assert premature_collapse(predicted_steps=1, gold_hops=2) == 1.0
    assert over_extension(predicted_steps=3, gold_hops=2) == 0.5
    assert step_alignment(predicted, gold) == 1.0
