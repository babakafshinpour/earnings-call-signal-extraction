"""
Convert raw transcripts + cleaned labels into JSONL training format.

Reads:
    data/raw/{transcript_id}.json           — scraped transcripts
    data/labels_cleaned/{transcript_id}.json — guidance labels

Writes:
    data/train.jsonl                        — 10 lines (training set)
    data/eval.jsonl                         — 3 lines (eval set)

Each line is one chat-formatted example:
    {"messages": [
        {"role": "system", "content": "<instructions + schema>"},
        {"role": "user", "content": "<CEO + CFO prepared remarks>"},
        {"role": "assistant", "content": "<JSON labels>"}
    ]}

Design choices baked in:
- Input scope = CEO + CFO prepared_remarks segments only (excludes IR/operator
  housekeeping and Q&A noise). Most labels come from this region.
- System prompt is compressed compared to the labeling prompt — the labels
  themselves carry most of the teaching signal during fine-tuning.
- One example per transcript. No chunking; we report any transcript that would
  exceed max_seq_length and let you decide whether to truncate or split.

Usage:
    python build_training_jsonl.py
    python build_training_jsonl.py --max-seq-len 4096 --report-overflow
"""

import argparse
import json
from pathlib import Path

RAW_DIR = Path("data/raw")
LABELS_DIR = Path("data/label")
OUT_DIR = Path("data")

# Train/eval split — locked in earlier and stored in docs/dataset_split.md.
# If you swapped CAT_Q1-2026 to training, this matches that decision.
EVAL_TRANSCRIPTS = ["MSFT_Q3-2026", "JPM_Q1-2026", "CAT_Q4-2024"]
TRAIN_TRANSCRIPTS = [
    "MSFT_Q2-2026",
    "JPM_Q2-2025", "JPM_Q4-2025",
    "CAT_Q4-2025", "CAT_Q1-2026",
    "COST_Q1-2026", "COST_Q2-2026", "COST_Q3-2026",
    "UNH_Q1-2026", "UNH_Q2-2025",
]

# Roles to keep when extracting prepared_remarks (skip IR housekeeping etc.)
KEEP_ROLES = {"ceo", "cfo", "other_exec"}

# Approximate token estimator: 1 token ≈ 4 characters for English+JSON.
# Good enough for overflow warnings; the trainer uses the real tokenizer.
def approx_tokens(text: str) -> int:
    return len(text) // 4


SYSTEM_PROMPT = """You extract forward-looking financial guidance from earnings call transcripts.

Output ONLY a JSON object of the form:
{"guidance": [<record>, ...]}

Each record has these fields:
  metric          - one of: revenue, eps_gaap, eps_adjusted, operating_income,
                    operating_margin, gross_margin, fcf, capex, segment_revenue, other
  metric_detail   - string or null (segment name, qualifier, "constant currency", etc.)
  period          - normalized: Q1-2026, FY2026, H1-2026, CY2026, or long_term
  direction       - raised, lowered, reiterated, introduced, withdrawn, narrowed
  new_low, new_high, new_point   - numeric or null
  old_low, old_high, old_point   - numeric or null
  unit            - usd_millions, usd_billions, usd, percent, count, other
  currency        - USD/EUR/etc. or null
  vs_consensus    - above, in_line, below, not_stated
  confidence_language - verbatim qualifier phrase or null
  source_span     - verbatim quote from transcript
  speaker         - ceo, cfo, other_exec, analyst, operator, unknown

Rules:
1. Every number in any numeric field must appear verbatim in source_span
   (or be arithmetically derivable from numbers in the span).
2. Do not infer numbers from qualitative phrases ("low double-digit growth"
   does NOT become 10-13).
3. Normalize dollar amounts to millions: $4.2B → 4200, $700M → 700.
4. Past-tense statements about the quarter just reported are not guidance.
5. Capital return / buyback / dividend statements are out of scope.
6. Planning assumptions and operational drivers cited as context for guidance
   are not labeled separately.
7. If the CFO restates guidance the CEO already gave earlier in the same call
   (e.g. "as [CEO] mentioned, we expect..."), do not create a second record.
8. For qualitative records with no numeric value, set numeric fields to null
   and use unit=percent for growth/comparison framings, dollar unit otherwise.

Return only the JSON object, no commentary."""


def load_transcript(transcript_id: str) -> dict:
    path = RAW_DIR / f"{transcript_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"raw transcript not found: {path}")
    return json.loads(path.read_text())


def load_labels(transcript_id: str) -> dict:
    path = LABELS_DIR / f"{transcript_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"labels not found: {path}")
    return json.loads(path.read_text())


def extract_input_text(transcript: dict) -> str:
    """
    Concatenate CEO + CFO + other_exec prepared_remarks segments into one
    block with speaker headers. If the participants section didn't parse
    (all roles are 'unknown'), fall back to including every prepared_remarks
    segment.
    """
    segments = transcript.get("segments", [])
    prepared = [s for s in segments if s.get("section") == "prepared_remarks"]
    kept = [s for s in prepared if s.get("role") in KEEP_ROLES]
    if not kept:
        # Fallback: participants didn't parse, take all prepared remarks.
        kept = prepared

    parts = []
    for s in kept:
        speaker = s.get("speaker", "Unknown")
        role = s.get("role", "unknown")
        text = s.get("text", "").strip()
        if not text:
            continue
        parts.append(f"[{speaker} — {role.upper()}]\n{text}")
    return "\n\n".join(parts)


def build_example(transcript_id: str) -> dict:
    transcript = load_transcript(transcript_id)
    labels = load_labels(transcript_id)

    user_text = extract_input_text(transcript)

    # The assistant output is JUST the guidance object — drop the metadata
    # (transcript_id, labeled_by, etc.) since the model shouldn't reproduce it.
    # Use compact JSON (no indent) to save ~30-40% of output tokens during
    # training. The model learns structure either way; guided decoding at
    # inference time enforces schema validity.
    assistant_obj = {"guidance": labels.get("guidance", [])}
    assistant_text = json.dumps(assistant_obj, separators=(",", ":"))

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def write_jsonl(transcript_ids: list, out_path: Path, max_seq_len: int) -> tuple:
    """Returns (n_examples, n_records, overflows)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overflows = []
    total_records = 0
    with open(out_path, "w") as f:
        for tid in transcript_ids:
            try:
                ex = build_example(tid)
            except FileNotFoundError as e:
                print(f"  SKIP {tid}: {e}")
                continue
            total_chars = sum(len(m["content"]) for m in ex["messages"])
            tokens = approx_tokens(json.dumps(ex))
            if tokens > max_seq_len:
                overflows.append((tid, tokens))
            n_recs = len(json.loads(ex["messages"][2]["content"])["guidance"])
            total_records += n_recs
            print(f"  {tid}: {n_recs} records, ~{tokens} tokens, "
                  f"{total_chars:,} chars")
            f.write(json.dumps(ex) + "\n")
    return total_records, overflows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seq-len", type=int, default=16384,
                    help="Token budget per example; reports overflow. "
                         "Default 16384 fits the densest CAT/MSFT transcripts "
                         "with margin. QLoRA on Qwen 9B + A100 40GB handles "
                         "this comfortably (~30GB VRAM at batch=1).")
    args = ap.parse_args()

    print(f"=== TRAIN ===")
    train_records, train_overflows = write_jsonl(
        TRAIN_TRANSCRIPTS, OUT_DIR / "train.jsonl", args.max_seq_len)

    print(f"\n=== EVAL ===")
    eval_records, eval_overflows = write_jsonl(
        EVAL_TRANSCRIPTS, OUT_DIR / "eval.jsonl", args.max_seq_len)

    print(f"\n=== SUMMARY ===")
    print(f"train.jsonl: {len(TRAIN_TRANSCRIPTS)} examples, "
          f"{train_records} total guidance records")
    print(f"eval.jsonl:  {len(EVAL_TRANSCRIPTS)} examples, "
          f"{eval_records} total guidance records")

    if train_overflows or eval_overflows:
        print(f"\nWARNING: {len(train_overflows) + len(eval_overflows)} "
              f"examples exceed --max-seq-len={args.max_seq_len}:")
        for tid, tok in train_overflows + eval_overflows:
            print(f"  {tid}: ~{tok} tokens")
        print(f"\nOptions: increase --max-seq-len during training, "
              f"or accept truncation (some labels at the tail may be lost).")
    else:
        print(f"\nAll examples fit within --max-seq-len={args.max_seq_len}. OK.")


if __name__ == "__main__":
    main()
