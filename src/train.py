"""
QLoRA fine-tuning of Qwen 2.5 7B Instruct on earnings call guidance extraction.

v3: switched DataCollatorForLanguageModeling → DataCollatorForSeq2Seq.
The previous collator was silently overwriting our masked labels with
input_ids.clone(), defeating the prompt masking entirely. The new collator
preserves pre-set labels and pads them with -100.

Reads:
    data/train.jsonl
    data/eval.jsonl

Writes:
    outputs/lora_adapters/
    outputs/checkpoints/
    outputs/training_log.json
"""

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ---- Config ----

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 16384
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

OUTPUT_DIR = Path("outputs")
TRAIN_FILE = Path("data/train.jsonl")
EVAL_FILE = Path("data/eval.jsonl")

QWEN_EOS_TOKEN = "<|im_end|>"
IGNORE_INDEX = -100


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.eos_token = QWEN_EOS_TOKEN
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(QWEN_EOS_TOKEN)
    if tokenizer.pad_token is None or tokenizer.pad_token == "<EOS_TOKEN>":
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"EOS token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    train_data = load_jsonl(TRAIN_FILE)
    eval_data = load_jsonl(EVAL_FILE)
    print(f"Loaded {len(train_data)} train + {len(eval_data)} eval examples")

    def tokenize_with_prompt_masking(ex):
        """Tokenize and mask labels on the system+user portion."""
        prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = tokenizer(
            prompt_text, truncation=True, max_length=MAX_SEQ_LENGTH,
            padding=False, return_tensors=None,
        )
        prompt_len = len(prompt_tokens["input_ids"])

        full_text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        full_tokens = tokenizer(
            full_text, truncation=True, max_length=MAX_SEQ_LENGTH,
            padding=False, return_tensors=None,
        )

        labels = list(full_tokens["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX
        full_tokens["labels"] = labels
        return full_tokens

    train_ds = Dataset.from_list(train_data).map(
        tokenize_with_prompt_masking, remove_columns=["messages"]
    )
    eval_ds = Dataset.from_list(eval_data).map(
        tokenize_with_prompt_masking, remove_columns=["messages"]
    )

    # Verify masking actually applied (count -100 vs real labels per example)
    n_train_tokens = sum(
        sum(1 for lbl in ex["labels"] if lbl != IGNORE_INDEX)
        for ex in train_ds
    )
    n_total_tokens = sum(len(ex["input_ids"]) for ex in train_ds)
    print(f"Training on {n_train_tokens:,} assistant tokens "
          f"out of {n_total_tokens:,} total ({100*n_train_tokens/n_total_tokens:.1f}%)")
    print(f"Per-example assistant lengths: "
          f"{[sum(1 for l in ex['labels'] if l != IGNORE_INDEX) for ex in train_ds][:5]}...")

    # ---- Collator: Seq2Seq preserves pre-set labels (LanguageModeling overwrote them) ----
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        per_device_eval_batch_size=1,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\n=== TRAINING ===")
    trainer_stats = trainer.train()

    adapter_dir = OUTPUT_DIR / "lora_adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nLoRA adapters saved to {adapter_dir}")

    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "train_runtime_sec": trainer_stats.metrics.get("train_runtime"),
            "train_loss": trainer_stats.metrics.get("train_loss"),
            "log_history": trainer.state.log_history,
        }, f, indent=2)
    print(f"Training log saved to {log_path}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()