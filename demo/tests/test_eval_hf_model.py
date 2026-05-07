from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_model


def test_tool_fragment_detects_unpaired_tool_tag_without_counting_valid_call() -> None:
    raw_prediction = "</tool_call>\n\n<tool_call>"

    tool_call_count, valid_tool_call_count = eval_hf_model.parse_tool_calls(raw_prediction)
    record = {
        "em": 0.0,
        "f1": 0.0,
        "answer_tag_present": False,
        "tool_call_count": tool_call_count,
        "valid_tool_call_count": valid_tool_call_count,
        "tool_fragment_present": eval_hf_model.has_tool_fragment(raw_prediction),
        "raw_prediction": raw_prediction,
    }
    summary = eval_hf_model.build_summary([record])

    assert tool_call_count == 0
    assert valid_tool_call_count == 0
    assert record["tool_fragment_present"] is True
    assert summary["tool_call_rate"] == 0.0
    assert summary["tool_fragment_rate"] == 1.0


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        mapping = {
            "<tool_call>": [151657],
            "</tool_call>": [151658],
        }
        return mapping[text]


def test_tool_tag_bad_words_ids_contains_open_and_close_tags() -> None:
    assert eval_hf_model.tool_tag_bad_words_ids(FakeTokenizer()) == [[151657], [151658]]
