# Earnings Call Signal Extraction

Fine-tune Qwen 2.5 7B Instruct (QLoRA) to extract structured forward-looking
guidance from earnings call transcripts and emit JSON. The model reads CEO + CFO
prepared remarks from a quarterly call and outputs one record per piece of
guidance (metric, period, direction, numeric range, unit, speaker, verbatim
source span, etc.).

## Pipeline

1. **Scrape** transcripts from Motley Fool → `data/raw/{TICKER}_{PERIOD}.json`
2. **Label** guidance records by hand → `data/label/{TICKER}_{PERIOD}.json`
3. **Build** chat-formatted JSONL → `data/train.jsonl`, `data/eval.jsonl`
4. **Train** QLoRA adapters on a single A100 → `outputs/lora_adapters/`
5. **Evaluate** the adapters on held-out transcripts → `outputs/eval_predictions.jsonl`

## Dataset

13 transcripts across 5 companies in 3 sectors (Microsoft, JPMorgan,
Caterpillar, Costco, UnitedHealth). 10 train / 3 eval. The eval set holds the
most recent quarter per sector to measure generalization to time periods
adjacent to training data. See [docs/dataset_split.md](docs/dataset_split.md).

## Layout

```
src/
  scrape_fool.py            scrape + parse Motley Fool transcripts
  build_training_jsonl.py   raw + labels → train.jsonl / eval.jsonl
  train.py                  QLoRA fine-tune on A100 (Unsloth + Qwen 2.5 7B)
  smoke_test.py             one-shot generation against the trained adapters
  eval_local.py             full eval on Mac (MPS) / CPU / CUDA, with resume
  code.sh                   GCP A100 VM launch + scp helpers
  urls.txt                  list of transcript URLs scraped
data/
  raw/                      scraped transcripts (gitignored)
  cache/                    HTTP cache (gitignored)
  label/                    hand-labeled guidance JSON
  train.jsonl, eval.jsonl   chat-formatted training examples
outputs/
  lora_adapters/            trained LoRA weights
  checkpoints/              intermediate Trainer checkpoints
  training_log.json         loss + metrics history
  eval_predictions.jsonl    eval generations (one line per transcript)
```

## Output schema

Each example targets a single JSON object:

```json
{"guidance": [
  {
    "metric": "revenue",
    "metric_detail": null,
    "period": "Q4-2026",
    "direction": "raised",
    "new_low": 4200, "new_high": 4300, "new_point": null,
    "old_low": null, "old_high": null, "old_point": null,
    "unit": "usd_millions",
    "currency": "USD",
    "vs_consensus": "not_stated",
    "confidence_language": "we now expect",
    "source_span": "...",
    "speaker": "cfo"
  }
]}
```

Constraints baked into the system prompt: every number in a numeric field must
appear verbatim in `source_span` (or be derivable from it); qualitative phrases
like "low double-digit growth" are not converted to numbers; dollar amounts are
normalized to millions. Full rules live in
[`src/build_training_jsonl.py`](src/build_training_jsonl.py).

## Run it

```bash
# 1. Scrape (rate-limited, HTML-cached)
python src/scrape_fool.py --urls src/urls.txt

# 2. Build training JSONL
python src/build_training_jsonl.py

# 3. Train on A100 (see src/code.sh for VM creation)
python src/train.py

# 4. Evaluate
python src/eval_local.py     # Mac / CPU / CUDA
python src/smoke_test.py     # GPU-only single-example check
```

## Training setup

- Base: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- LoRA: r=32, alpha=64, dropout=0, targets `q/k/v/o_proj`
- 5 epochs, lr=2e-4 cosine with 10% warmup, batch=1, grad accum=4
- Max seq len 16384 (covers the densest CAT/MSFT transcripts)
- Prompt-masked labels via `DataCollatorForSeq2Seq` (the LM collator silently
  overwrites pre-set labels, defeating the masking)
