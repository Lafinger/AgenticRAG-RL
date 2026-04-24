from __future__ import annotations

from agentic_rag_rl.evaluation import exact_match, hop_recall, over_extension, premature_collapse, step_alignment, token_f1


def test_exact_match_and_f1() -> None:
    assert exact_match("双水村", "双水村") == 1.0
    assert token_f1("东拉河和哭咽河", "东拉河和哭咽河") == 1.0
    assert token_f1("东拉河", "孙少平") == 0.0


def test_hop_aware_metrics() -> None:
    predicted = ["corpus_chunkids_000001", "corpus_chunkids_000002"]
    gold = ["corpus_chunkids_000001", "corpus_chunkids_000002"]

    assert hop_recall(predicted, gold) == 1.0
    assert premature_collapse(predicted_steps=1, gold_hops=2) == 1.0
    assert over_extension(predicted_steps=3, gold_hops=2) == 0.5
    assert step_alignment(predicted, gold) == 1.0
