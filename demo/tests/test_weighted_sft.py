from __future__ import annotations

import torch
import pytest

from training.sft_label_mask import IGNORE_INDEX
from training.weighted_sft import WeightedDataCollatorForLanguageModeling, weighted_causal_lm_loss


def test_weighted_data_collator_pads_loss_weights() -> None:
    collator = WeightedDataCollatorForLanguageModeling(pad_token_id=0)

    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [IGNORE_INDEX, 2], "loss_weights": [0.0, 12.0]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3], "loss_weights": [4.0]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 2], [3, IGNORE_INDEX]]
    assert batch["loss_weights"].tolist() == [[0.0, 12.0], [4.0, 0.0]]


def test_weighted_causal_lm_loss_uses_loss_weights() -> None:
    logits = torch.zeros((1, 3, 4), dtype=torch.float32)
    logits[0, 0, 1] = 8.0
    logits[0, 1, 2] = 8.0
    labels = torch.tensor([[IGNORE_INDEX, 1, 2]], dtype=torch.long)
    low_weight = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    high_weight = torch.tensor([[0.0, 1.0, 12.0]], dtype=torch.float32)

    low_loss = weighted_causal_lm_loss(logits, labels, low_weight)
    high_loss = weighted_causal_lm_loss(logits, labels, high_weight)

    assert high_loss.item() == pytest.approx(low_loss.item())

    bad_logits = logits.clone()
    bad_logits[0, 1, 2] = -8.0
    assert weighted_causal_lm_loss(bad_logits, labels, high_weight) > weighted_causal_lm_loss(bad_logits, labels, low_weight)
