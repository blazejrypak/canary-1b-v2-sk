# Project: Canary-1B-v2 Slovak Fine-tuning

You are helping me set up a fine-tuning project for NVIDIA's Canary-1B-v2 ASR
model on Slovak parliamentary speech, then benchmarking it on standard Slovak
ASR test sets. Training will happen on RunPod (cloud GPU); this local repo holds
only code, configs, and docs — never data, models, or secrets.

## Goal
Fine-tune `nvidia/canary-1b-v2` on the SloPalSpeech dataset
(`NaiveNeuron/SloPalSpeech`, ~2806h Slovak parliamentary speech, 16kHz, parquet
with embedded audio arrays) and report WER on:
  - CommonVoice 21 SK test split (`mozilla-foundation/common_voice_21_0`, "sk")
  - FLEURS SK test split (`google/fleurs`, "sk_sk")

Target: WER improvement on CV21 SK over the pretrained baseline, while tracking
domain-shift effects on FLEURS (parliamentary speech is a narrow domain;
FLEURS WER may rise — that's an expected, reportable result, not a bug).

## Local development (Mac)

No GPU needed for local work — the unit tests cover all pure-Python logic.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install package
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

> **Note:** The CLI scripts (`scripts/transform_slopal.py`, `scripts/benchmark.py`) require the full RunPod environment and cannot be run locally — they import `datasets`, `lhotse`, and `torch` which are not in `requirements-dev.txt`. Run them only on the pod.

These tests cover text normalisation, audio filtering rules, stratified sampling, and train/dev/test split logic. They run in <5s with no downloads.

## Key constraints — read before generating code
1. Canary-1B-v2 NATIVELY supports Slovak (it's one of 25 languages from the
   Granary pretraining set). Do NOT reinitialize the tokenizer, decoder, or
   pre-encode output layer — this is a domain adaptation task, not new-language
   addition. Reuse all pretrained weights.
2. Canary v2 requires per-sample prompt fields: `source_lang`, `target_lang`,
   `task`, `pnc`. For our case all four are `{"sk", "sk", "asr", "yes"}`.
   Missing these = the model doesn't know what to do.
3. Audio handling: SloPalSpeech audio arrays are already 16kHz mono float32 —
   no resampling needed. Segments are ~25s avg; filter out <1s or >40s
   (Canary encoder limit).
4. Text normalization: KEEP punctuation and capitalization for training (Canary
   supports PnC). Only Unicode NFC normalization + whitespace collapse. For
   WER computation in benchmarks, apply identical normalization to both
   references and hypotheses (lowercase, strip punctuation), matching the
   open-asr-leaderboard convention.
5. Data format: use Lhotse Shar shards (not NeMo JSON manifest) for training.
   Canary v2 has first-class Lhotse support via `use_lhotse: true`, and Shar
   gives sequential I/O which matters on RunPod network volumes. JSON manifest
   only for tiny (<50h) sanity-check runs.
6. Training strategy: start with a stratified ~500h subset (sampled
   proportionally across `snapshot` years) rather than full 2806h. Diminishing
   returns past 500h for domain adaptation; cost goes from ~$30 to ~$400+.

## Runtime environment — important
- LOCAL (this repo): code only. NO datasets, NO `.nemo`/`.ckpt` checkpoints, NO
  audio files, NO `.env` with real tokens. `.gitignore` must exclude these.
- REMOTE (RunPod pod): `nvcr.io/nvidia/nemo:25.04` container, A100 SXM 80GB or
  H100. `/workspace` is a persistent Network Volume (~200GB).
- Repo will be cloned to pod via private GitHub repo + read-only deploy key.
- Secrets (HF_TOKEN, etc.) come from RunPod environment variables, NEVER from
  committed files. The repo contains `.env.example` only.

## Files I want in this repo
- `src/canary_sk/normalize.py` — `normalize_text()` (training, keeps PnC) and `normalize_for_wer()` (benchmark evaluation, lowercase + strip punct).
- `src/canary_sk/transform.py` — all SloPalSpeech conversion logic: `row_to_cut`, `is_valid`, `stratified_indices`, `split_cuts`. Lhotse imported lazily inside `row_to_cut` so the module is importable on Mac.
- `src/canary_sk/benchmark.py` — `evaluate_dataset()` used by the benchmark CLI. Only runs on RunPod (requires torch, soundfile, jiwer).
- `scripts/transform_slopal.py` — thin CLI wrapper: argparse + calls `canary_sk.transform`.
- `scripts/benchmark.py` — thin CLI wrapper: argparse + calls `canary_sk.benchmark`. Imports NeMo lazily.
- `tests/` — unit tests for all pure-Python logic. Run with `pytest tests/`.
- `pyproject.toml` — makes `canary_sk` importable as a package; configures pytest.
- `finetune.yaml` — NeMo config: `init_from_pretrained_model: nvidia/canary-1b-v2`, Lhotse Shar dataloaders with dynamic bucketing, AdamW lr=1e-5 (low — adaptation, not pretraining), WarmupAnnealing 1000 steps, bf16-mixed, SpecAugment, max_steps=15000, val every 1000 steps monitoring val_wer.
- `requirements.txt` — RunPod full stack: `nemo_toolkit[asr]>=2.0`, `lhotse>=1.30`, `jiwer>=3.0`, `datasets>=2.20`, `soundfile`.
- `requirements-dev.txt` — Mac lightweight: `numpy>=1.26`, `pytest>=8.0`.
- `.gitignore` — excludes `*.nemo`, `*.ckpt`, `*.pt`, `wavs/`, `slopal_lhotse/`, `exp/`, `.env`, `*.log`, `__pycache__/`.
- `.env.example` — placeholder for `HF_TOKEN=hf_xxxxx`.
- `README.md` — this file.

## Style guidelines
- Python: type hints where they add clarity, not everywhere. Docstrings on
  top-level functions only. argparse for CLI scripts. Print progress with
  enough info to debug (counts, durations, ETAs) but not so much it spams.
- YAML: comments above non-obvious values explaining WHY (e.g., why lr=1e-5,
  why bf16-mixed, why batch_duration=360).
- Keep scripts standalone — each runnable on a fresh pod with just the repo
  cloned + `pip install -r requirements.txt`. No hidden cross-file state.
- No premature abstractions. Two scripts that share 10 lines of normalization
  can each have their own copy; don't build a `utils/` for three weeks.

## When you suggest changes
- Verify against current NeMo API. NeMo's ASR API has changed across versions;
  config keys, class paths (`EncDecMultiTaskModel`), and Lhotse integration
  details from older docs may be wrong. If unsure about a specific API call,
  flag it rather than guess.
- Flag any assumption you're making about the dataset schema. The HF dataset
  card and the parquet column names are the source of truth, not my
  description.
- Cost-impacting suggestions (more epochs, bigger subset, bigger GPU) must
  come with an estimated $ delta. Budget is ~$50 total.

## Quickstart (RunPod)

**Pod setup (one-time):**

```bash
# 1. Clone the repo (replace with your GitHub URL)
git clone git@github.com:<your-org>/canary-1b-v2-sk.git /workspace/canary-1b-v2-sk
cd /workspace/canary-1b-v2-sk

# 2. Install dependencies (lhotse, jiwer, datasets on top of the NeMo base image)
pip install -r requirements.txt
```

**Run the pipeline:**

```bash
# 1. Transform dataset (run once, ~2h on A100)
python scripts/transform_slopal.py \
    --output-dir /workspace/slopal_lhotse \
    --target-hours 500

# 2. Fine-tune (~6–8h on A100 at max_steps=15000)
python -m nemo.collections.asr.scripts.speech_to_text_finetune \
    --config-path=/workspace/canary-1b-v2-sk \
    --config-name=finetune

# 3. Benchmark (pretrained baseline vs fine-tuned)
python scripts/benchmark.py --pretrained nvidia/canary-1b-v2
# List available checkpoints: ls /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/
python scripts/benchmark.py \
    --model /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best-checkpoint>.nemo
```

Expected cost at current RunPod A100 rates (~$2.50/h): ~$30 total.
