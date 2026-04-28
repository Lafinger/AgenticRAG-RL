from __future__ import annotations

from multiprocessing import freeze_support


def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    import torch
    from unsloth import FastLanguageModel, FastModel
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    del FastModel

    print("torch", torch.__version__, torch.version.cuda, torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))

    max_seq_length = 512
    url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
    dataset = load_dataset("json", data_files={"train": url}, split="train[:128]")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        full_finetuning=False,
        trust_remote_code=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=SFTConfig(
            max_seq_length=max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            warmup_steps=0,
            max_steps=1,
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_8bit",
            seed=3407,
            dataset_num_proc=None,
            report_to=[],
        ),
    )
    trainer.train()
    print("unsloth smoke train ok")


if __name__ == "__main__":
    freeze_support()
    main()
