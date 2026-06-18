"""
Evaluate the fine-tuned model on held-out eval transcripts.
Runs on Mac (MPS), CPU, or CUDA.

v2: aggressive memory cleanup between examples (MPS leaks the KV cache
on Mac, causing OOM on the third example otherwise). Also saves results
incrementally so partial progress survives a crash.

Reads:
    data/eval.jsonl
    outputs/lora_adapters/

Writes:
    outputs/eval_predictions.jsonl  (appended one line per example)
"""

import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = Path("outputs/lora_adapters")
EVAL_FILE = Path("data/eval.jsonl")
OUTPUT_FILE = Path("outputs/eval_predictions.jsonl")

MAX_NEW_TOKENS = 8192

# ---- Device selection ----
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Device: {DEVICE}, dtype: {DTYPE}")


def free_memory():
    """Aggressively free memory between generations."""
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def already_done_idxs(path: Path) -> set:
    """Read existing output file to skip already-processed examples."""
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["example_idx"])
            except Exception:
                continue
    return done


def main():
    # ---- Load base model + tokenizer ----
    print(f"\nLoading tokenizer from {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading base model {BASE_MODEL} (~14GB; may take a few minutes)...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ---- Load LoRA adapters ----
    print(f"\nLoading LoRA adapters from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    model.eval()
    print(f"  Adapters loaded.")

    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # ---- Resume support: skip already-done examples ----
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = already_done_idxs(OUTPUT_FILE)
    if done:
        print(f"\nFound {len(done)} examples already done in {OUTPUT_FILE}; "
              f"skipping those.")

    # ---- Load eval examples ----
    eval_data = [json.loads(l) for l in open(EVAL_FILE)]
    print(f"\nLoaded {len(eval_data)} eval transcripts.")

    # ---- Run inference, saving incrementally ----
    for i, example in enumerate(eval_data):
        if i in done:
            print(f"\nSkipping example {i+1}/{len(eval_data)} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"Example {i+1}/{len(eval_data)}")
        print(f"{'='*60}")

        # Aggressive cleanup before each generation
        free_memory()

        prompt_messages = [m for m in example["messages"] if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        print(f"  Prompt length: {prompt_len} tokens")

        print(f"  Generating (max {MAX_NEW_TOKENS} new tokens)...")
        t0 = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
                use_cache=True,
            )
        gen_time = time.time() - t0
        new_tokens = output[0][prompt_len:].clone().cpu()  # move off device immediately
        n_new = len(new_tokens)
        print(f"  Generated {n_new} tokens in {gen_time:.1f}s "
              f"({n_new/max(gen_time,0.01):.1f} tok/s)")

        # Decode
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Build result record
        ground_truth = json.loads(example["messages"][-1]["content"])
        gt_n_records = len(ground_truth.get("guidance", []))

        result = {
            "example_idx": i,
            "prompt_tokens": prompt_len,
            "generated_tokens": n_new,
            "generation_time_sec": gen_time,
            "ground_truth_records": gt_n_records,
            "raw_output": generated_text,
        }
        try:
            parsed = json.loads(generated_text)
            n_pred = len(parsed.get("guidance", []))
            result["valid_json"] = True
            result["predicted_records"] = n_pred
            result["prediction"] = parsed
            print(f"  VALID JSON. Predicted {n_pred} records, truth has {gt_n_records}.")
        except json.JSONDecodeError as e:
            result["valid_json"] = False
            result["predicted_records"] = None
            result["json_error"] = str(e)
            print(f"  INVALID JSON ({e}); truth has {gt_n_records} records.")
        result["ground_truth"] = ground_truth

        # Append immediately so partial progress survives a crash
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"  Saved to {OUTPUT_FILE}")

        # Free everything possible before next example
        del output, inputs, new_tokens
        free_memory()

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"=== SUMMARY ===")
    print(f"{'='*60}")
    with open(OUTPUT_FILE) as f:
        all_results = [json.loads(l) for l in f]
    for r in sorted(all_results, key=lambda r: r["example_idx"]):
        status = "OK" if r["valid_json"] else "INVALID"
        n_pred = r.get("predicted_records", "?")
        print(f"  ex{r['example_idx']}: {status:7} | "
              f"pred={n_pred} truth={r['ground_truth_records']} | "
              f"{r['generated_tokens']} tokens in {r['generation_time_sec']:.0f}s")


if __name__ == "__main__":
    main()