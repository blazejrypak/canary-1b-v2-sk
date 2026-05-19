# Design: Canary-1B-v2-SK Project Structure Refactor

**Date:** 2026-05-19  
**Status:** Approved  
**Scope:** Refactor flat scripts into standard Python package, add local unit tests, set up uv environment for Mac

---

## Goal

Reorganise the existing 3-file flat repo into a proper Python package layout so that:
1. Logic is testable on a MacBook without GPU, NeMo, or HF downloads
2. The shared normalisation code (currently duplicated) lives in one place
3. CLI scripts become thin wrappers that can be called on RunPod unchanged

---

## Project Structure

```
canary-1b-v2-sk/
├── src/
│   └── canary_sk/
│       ├── __init__.py          # empty
│       ├── normalize.py         # normalize_text() + normalize_for_wer()
│       ├── transform.py         # row_to_cut, is_valid, stratified_indices, split_cuts
│       └── benchmark.py         # evaluate_dataset()
├── scripts/
│   ├── transform_slopal.py      # CLI: argparse + main(), imports canary_sk.transform
│   └── benchmark.py             # CLI: argparse + main(), imports canary_sk.benchmark
├── tests/
│   ├── conftest.py              # shared fixtures: fake_audio (1s 16kHz sine)
│   ├── test_normalize.py        # unit tests for normalize_text + normalize_for_wer
│   └── test_transform.py        # unit tests for is_valid, split math, stratified sampling
├── finetune.yaml                # unchanged (NeMo training config)
├── pyproject.toml               # package config; pytest config
├── requirements.txt             # RunPod: nemo_toolkit[asr]>=2.0, lhotse>=1.30, jiwer>=3.0, datasets>=2.20, soundfile
├── requirements-dev.txt         # Mac: numpy>=1.26, pytest>=8.0, jiwer>=3.0, unicodedata2>=15.0
├── .env.example                 # HF_TOKEN=hf_xxxxx
├── .gitignore                   # *.nemo, *.ckpt, *.pt, exp/, slopal_lhotse/, wavs/, .env, *.log
└── README.md                    # updated: uv setup, 3 commands, expected cost/time
```

---

## Key Refactor: `normalize.py`

**Problem:** `normalize_text` in `transform_slopal_to_lhotse.py` and `normalize` in `benchmark.py` are two separate implementations of similar logic. If one is updated the other can silently diverge.

**Solution:** `src/canary_sk/normalize.py` exports two clearly named functions:

```python
def normalize_text(text: str) -> str:
    """Training normalization: NFC + whitespace collapse only. Keeps punctuation and capitalisation (PnC)."""

def normalize_for_wer(text: str) -> str:
    """Benchmark normalization: NFC + lowercase + strip punctuation. Applied identically to refs and hyps."""
```

Both CLI scripts import from here. One place to change.

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "canary-sk"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.hatch.build.targets.wheel]
packages = ["src/canary_sk"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Runtime dependencies stay in `requirements.txt` (RunPod controls installation). Dev deps in `requirements-dev.txt`.

---

## Environment Setup (Mac, uv)

```bash
# One-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual env with Python 3.11
uv venv --python 3.11
source .venv/bin/activate

# Install package in editable mode + dev deps
uv pip install -e .
uv pip install -r requirements-dev.txt

# Run tests (no GPU, no NeMo, no HF)
pytest tests/
```

`requirements-dev.txt`:
```
numpy>=1.26
pytest>=8.0
jiwer>=3.0
unicodedata2>=15.0
```

---

## Tests

All tests run on Mac with no GPU, no HF downloads, no NeMo import.

### `tests/conftest.py`
- `fake_audio` fixture: 1-second 16kHz sine wave as float32 numpy array

### `tests/test_normalize.py`
| Input | Function | Expected output |
|---|---|---|
| `"Ahoj  svět !"` | `normalize_text` | `"Ahoj svět !"` (whitespace collapsed, PnC kept) |
| `"café"` | `normalize_text` | `"café"` (NFC applied) |
| `"Ahoj, Svet!"` | `normalize_for_wer` | `"ahoj svet"` (lowercase + no punct) |
| `"  Čo  je  to?"` | `normalize_for_wer` | `"čo je to"` (Slovak chars survive) |

### `tests/test_transform.py`
| Input | Function | Expected |
|---|---|---|
| `duration=0.5, text="hello world"` | `is_valid` | `False` (too short) |
| `duration=1.0, text="hello world"` | `is_valid` | `True` (boundary) |
| `duration=40.0, text="x"*200` | `is_valid` | `True` (boundary) |
| `duration=40.1, text="hello"` | `is_valid` | `False` (too long) |
| `duration=10.0, text="hi"` | `is_valid` | `False` (cps < 3) |
| `duration=1.0, text="x"*31` | `is_valid` | `False` (cps > 30) |
| `meta=[], target_hours=0` | `stratified_indices` | empty set |
| `meta=full, target_hours=very_large` | `stratified_indices` | all indices |
| `cuts=N cuts, dev=1h, test=1h` | `split_cuts` | dev≥1h, test≥1h, no overlap |

---

## Scripts (CLI wrappers)

`scripts/transform_slopal.py` — thin wrapper, ~20 lines:
```python
from canary_sk.transform import row_to_cut, stratified_indices, split_cuts
# argparse + main() only; normalize_text is called internally by row_to_cut
```

`scripts/benchmark.py` — thin wrapper, ~30 lines:
```python
from canary_sk.benchmark import evaluate_dataset
# argparse + main() only; normalize_for_wer is called internally by evaluate_dataset
```

---

## What Does NOT Change

- `finetune.yaml` — untouched, NeMo config is correct
- All RunPod CLI commands from README — same arguments, same behaviour
- All business logic — no algorithm changes, pure structural move

---

## Out of Scope

- No new features
- No CI/CD setup
- No type annotation overhaul
- No change to finetune.yaml NeMo config
