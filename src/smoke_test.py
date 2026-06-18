"""
Quick smoke test: load the trained model + LoRA adapters, run inference on
one eval example, print the result for visual inspection.

Usage:
    python smoke_test.py
"""
import json
from pathlib import Path
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 16384
ADAPTER_DIR = "outputs/lora_adapters"
EVAL_FILE = "data/eval.jsonl"

# Load base model
print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
tokenizer.eos_token = "<|im_end|>"
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

# Load LoRA adapters on top of base model
print("Loading LoRA adapters...")
model.load_adapter(ADAPTER_DIR)
FastLanguageModel.for_inference(model)  # enables fast inference path

# Load first eval example
with open(EVAL_FILE) as f:
    example = json.loads(f.readline())

# Prepare prompt: system + user only (no assistant — we want model to generate it)
prompt_messages = [m for m in example["messages"] if m["role"] != "assistant"]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True
)
print(f"\nPrompt length: {len(tokenizer.encode(prompt_text))} tokens")

# Generate
print("\nGenerating...")
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,           # greedy
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only the new tokens (after the prompt)
generated_tokens = output[0][inputs["input_ids"].shape[1]:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n=== GENERATED OUTPUT ===")
print(generated_text)
print("\n=== END ===")

# Try to parse as JSON
print("\n=== JSON VALIDITY CHECK ===")
try:
    parsed = json.loads(generated_text)
    n_records = len(parsed.get("guidance", []))
    print(f"VALID JSON. {n_records} guidance records extracted.")
    if n_records > 0:
        print(f"\nFirst record:\n{json.dumps(parsed['guidance'][0], indent=2)}")
except json.JSONDecodeError as e:
    print(f"INVALID JSON: {e}")
    print(f"First 500 chars of output: {generated_text[:500]}")

# Compare to ground truth
print("\n=== GROUND TRUTH ===")
truth = json.loads(example["messages"][-1]["content"])
print(f"Ground truth has {len(truth['guidance'])} guidance records.")