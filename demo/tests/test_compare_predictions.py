from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import compare_predictions


def test_compare_predictions_rejects_question_mismatch_by_default() -> None:
    base = compare_predictions.index_by_question([{"question": "问题A", "prediction": "甲", "gold": "甲"}])
    sft = compare_predictions.index_by_question([{"question": "问题B", "prediction": "乙", "gold": "乙"}])

    with pytest.raises(SystemExit, match="question 集合不一致"):
        compare_predictions.validate_question_alignment(base, sft, allow_mismatch=False)


def test_compare_predictions_allows_question_mismatch_when_explicit() -> None:
    base = compare_predictions.index_by_question([{"question": "问题A", "prediction": "甲", "gold": "甲"}])
    sft = compare_predictions.index_by_question([{"question": "问题B", "prediction": "乙", "gold": "乙"}])

    base_only, sft_only = compare_predictions.validate_question_alignment(base, sft, allow_mismatch=True)

    assert base_only == {"问题A"}
    assert sft_only == {"问题B"}


def test_compare_predictions_summarizes_tool_fragment_rate() -> None:
    records = [
        compare_predictions.enrich(
            {
                "question": "问题",
                "prediction": "</tool_call>",
                "raw_prediction": "</tool_call>",
                "gold": "答案",
            }
        )
    ]

    summary = compare_predictions.summarize(records)

    assert records[0]["tool_fragment_present"] is True
    assert summary["tool_fragment_rate"] == 1.0
