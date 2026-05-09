from __future__ import annotations

from typing import Any

from training.sft_label_mask import IGNORE_INDEX


class WeightedDataCollatorForLanguageModeling:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        attention_mask = [torch.tensor(example.get("attention_mask", [1] * len(example["input_ids"])), dtype=torch.long) for example in examples]
        labels = [torch.tensor(example["labels"], dtype=torch.long) for example in examples]
        loss_weights = [
            torch.tensor(example.get("loss_weights", [1.0 if label != IGNORE_INDEX else 0.0 for label in example["labels"]]), dtype=torch.float32)
            for example in examples
        ]

        return {
            "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX),
            "loss_weights": pad_sequence(loss_weights, batch_first=True, padding_value=0.0),
        }


def weighted_causal_lm_loss(logits: Any, labels: Any, loss_weights: Any) -> Any:
    import torch

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_weights = loss_weights[..., 1:].to(shift_logits.device).contiguous()
    valid_mask = shift_labels != IGNORE_INDEX

    labels_for_loss = shift_labels.clone()
    labels_for_loss[~valid_mask] = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels_for_loss.view(-1),
    ).view_as(shift_labels)

    weighted_mask = shift_weights * valid_mask.to(shift_weights.dtype)
    denom = weighted_mask.sum().clamp_min(1.0)
    return (token_loss * weighted_mask).sum() / denom


def make_weighted_sft_trainer_class(base_trainer_class: type[Any]) -> type[Any]:
    class WeightedSFTTrainer(base_trainer_class):  # type: ignore[misc, valid-type]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            loss_weights = inputs.pop("loss_weights", None)
            if loss_weights is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            labels = inputs["labels"]
            outputs = model(**inputs)
            loss = weighted_causal_lm_loss(outputs.logits, labels, loss_weights)
            return (loss, outputs) if return_outputs else loss

    return WeightedSFTTrainer
