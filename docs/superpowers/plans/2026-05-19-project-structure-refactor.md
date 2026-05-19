# Canary-SK Project Structure Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor 3 flat Python scripts into a standard Python package with unit-testable logic and a Mac-compatible uv dev environment.

**Architecture:** Extract shared logic into `src/canary_sk/` (3 modules: normalize, transform, benchmark), keep CLI entry points as thin wrappers in `scripts/`, add pytest tests covering all pure-Python functions — runnable on Mac with no GPU, no NeMo, no HF downloads.

**Tech Stack:** Python 3.11, uv, pytest, hatchling (build backend), numpy, jiwer

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/canary_sk/__init__.py` | Create | Empty — makes canary_sk a package |
| `src/canary_sk/normalize.py` | Create | `normalize_text()` + `normalize_for_wer()` |
| `src/canary_sk/transform.py` | Create | `is_valid`, `stratified_indices`, `split_cuts`, `row_to_cut` |
| `src/canary_sk/benchmark.py` | Create | `evaluate_dataset()` |
| `scripts/transform_slopal.py` | Create | CLI wrapper — argparse + calls canary_sk.transform |
| `scripts/benchmark.py` | Create | CLI wrapper — argparse + calls canary_sk.benchmark |
| `tests/conftest.py` | Create | `fake_audio` fixture |
| `tests/test_normalize.py` | Create | Unit tests for both normalize functions |
| `tests/test_transform.py` | Create | Unit tests for is_valid, stratified_indices, split_cuts |
| `pyproject.toml` | Create | Package config + pytest config |
| `requirements.txt` | Create | RunPod full stack |
| `requirements-dev.txt` | Create | Mac: numpy + pytest only |
| `.env.example` | Create | Placeholder for HF_TOKEN |
| `.gitignore` | Create | Excludes data, models, secrets, caches |
| `README.md` | Modify | Add uv local setup section, update commands |
| `transform_slopal_to_lhotse.py` | Delete | Replaced by scripts/transform_slopal.py + src/canary_sk/transform.py |
| `benchmark.py` (root) | Delete | Replaced by scripts/benchmark.py + src/canary_sk/benchmark.py |

---

## Task 1: Scaffold project structure

**Files:**
- Create: `src/canary_sk/__init__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `.gitignore`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/canary_sk scripts tests
```

- [ ] **Step 2: Create `src/canary_sk/__init__.py`**

```python
```
(Empty file — just needs to exist)

- [ ] **Step 3: Create `pyproject.toml`**

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

- [ ] **Step 4: Create `requirements.txt`** (RunPod full stack)

```
nemo_toolkit[asr]>=2.0
lhotse>=1.30
jiwer>=3.0
datasets>=2.20
soundfile
```

- [ ] **Step 5: Create `requirements-dev.txt`** (Mac — no NeMo, no lhotse, no GPU)

```
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 6: Create `.env.example`**

```
HF_TOKEN=hf_xxxxx
```

- [ ] **Step 7: Create `.gitignore`**

```
# Model checkpoints — never commit
*.nemo
*.ckpt
*.pt

# Experiment outputs
exp/

# Data
slopal_lhotse/
wavs/

# Secrets
.env

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/

# Logs
*.log
```

- [ ] **Step 8: Install package in editable mode (sets up imports)**

First, if you haven't installed uv yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment and install:
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements-dev.txt
```

- [ ] **Step 9: Verify import works**

```bash
python -c "import canary_sk; print('OK')"
```

Expected: `OK`

- [ ] **Step 10: Commit**

```bash
git add src/canary_sk/__init__.py pyproject.toml requirements.txt requirements-dev.txt .env.example .gitignore
git commit -m "feat: scaffold canary-sk package structure"
```

---

## Task 2: normalize.py — TDD

**Files:**
- Create: `tests/test_normalize.py`
- Create: `src/canary_sk/normalize.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import numpy as np
import pytest


@pytest.fixture
def fake_audio():
    """1-second 16 kHz sine wave — usable as a stand-in audio array in tests."""
    t = np.linspace(0, 1, 16000)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)
```

- [ ] **Step 2: Write failing tests in `tests/test_normalize.py`**

```python
from canary_sk.normalize import normalize_for_wer, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("Ahoj  svět  !") == "Ahoj svět !"


def test_normalize_text_keeps_punctuation():
    assert normalize_text("Ahoj, svet!") == "Ahoj, svet!"


def test_normalize_text_nfc():
    # café as NFD (e + combining accent) must become NFC (é)
    nfd = "café"
    assert normalize_text(nfd) == "café"


def test_normalize_text_nbsp():
    assert normalize_text("hello\u00a0world") == "hello world"


def test_normalize_for_wer_lowercase():
    assert normalize_for_wer("Ahoj Svet") == "ahoj svet"


def test_normalize_for_wer_strips_punctuation():
    assert normalize_for_wer("Ahoj, Svet!") == "ahoj svet"


def test_normalize_for_wer_keeps_slovak_chars():
    assert normalize_for_wer("Čo je to?") == "čo je to"


def test_normalize_for_wer_collapses_whitespace():
    assert normalize_for_wer("  Čo  je  to?") == "čo je to"
```

- [ ] **Step 3: Run tests — expect ImportError**

```bash
pytest tests/test_normalize.py -v
```

Expected: `ImportError: cannot import name 'normalize_for_wer' from 'canary_sk.normalize'` (module doesn't exist yet)

- [ ] **Step 4: Create `src/canary_sk/normalize.py`**

```python
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s'áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ]", re.UNICODE)


def normalize_text(text: str) -> str:
    """Training normalization: NFC + whitespace collapse. Keeps punctuation and capitalisation."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_wer(text: str) -> str:
    """Benchmark normalization: lowercase + strip punctuation. Apply to both ref and hyp."""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = text.replace("\u00a0", " ")
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
pytest tests/test_normalize.py -v
```

Expected:
```
tests/test_normalize.py::test_normalize_text_collapses_whitespace PASSED
tests/test_normalize.py::test_normalize_text_keeps_punctuation PASSED
tests/test_normalize.py::test_normalize_text_nfc PASSED
tests/test_normalize.py::test_normalize_text_nbsp PASSED
tests/test_normalize.py::test_normalize_for_wer_lowercase PASSED
tests/test_normalize.py::test_normalize_for_wer_strips_punctuation PASSED
tests/test_normalize.py::test_normalize_for_wer_keeps_slovak_chars PASSED
tests/test_normalize.py::test_normalize_for_wer_collapses_whitespace PASSED
8 passed
```

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_normalize.py src/canary_sk/normalize.py
git commit -m "feat: add normalize module with TDD tests"
```

---

## Task 3: transform.py — TDD for pure functions

**Files:**
- Create: `tests/test_transform.py`
- Create: `src/canary_sk/transform.py`

> Note: `row_to_cut` uses lhotse and is not tested locally. `is_valid`, `stratified_indices`, and `split_cuts` are pure Python and fully testable.

- [ ] **Step 1: Write failing tests in `tests/test_transform.py`**

```python
from types import SimpleNamespace

from canary_sk.transform import is_valid, split_cuts, stratified_indices


# ---- is_valid ----

def test_is_valid_too_short():
    assert is_valid(0.5, "hello world") is False


def test_is_valid_boundary_min():
    assert is_valid(1.0, "hello world") is True


def test_is_valid_too_long():
    assert is_valid(40.1, "hello world") is False


def test_is_valid_boundary_max():
    # 200 chars / 40s = 5 cps — within the 3–30 range
    assert is_valid(40.0, "x" * 200) is True


def test_is_valid_cps_too_low():
    # 2 chars / 10s = 0.2 cps
    assert is_valid(10.0, "hi") is False


def test_is_valid_cps_too_high():
    # 31 chars / 1s = 31 cps
    assert is_valid(1.0, "x" * 31) is False


def test_is_valid_text_too_short():
    # fewer than 5 chars always fails
    assert is_valid(5.0, "hi") is False


# ---- stratified_indices ----

def _make_meta(n: int, duration: float = 5.0, year: str = "2020") -> list:
    return [{"snapshot": year, "duration": duration} for _ in range(n)]


def test_stratified_zero_hours():
    meta = _make_meta(100)
    assert stratified_indices(meta, target_hours=0.0) == set()


def test_stratified_large_target_selects_all():
    meta = _make_meta(10, duration=5.0)  # 50s total
    result = stratified_indices(meta, target_hours=999.0)
    assert result == set(range(10))


def test_stratified_empty_input():
    assert stratified_indices([], target_hours=10.0) == set()


def test_stratified_proportional():
    meta = _make_meta(100, duration=36.0)  # 100 * 36s = 1h total
    result = stratified_indices(meta, target_hours=0.5)
    # 0.5h / 1h = 50% → roughly 50 items
    assert 40 <= len(result) <= 60


# ---- split_cuts ----

def _make_cuts(n: int, duration: float = 10.0) -> list:
    return [SimpleNamespace(id=f"cut_{i}", duration=duration) for i in range(n)]


def test_split_no_overlap():
    cuts = _make_cuts(200, duration=10.0)  # 2000s total
    train, dev, test = split_cuts(cuts, dev_hours=0.1, test_hours=0.1)
    train_ids = {c.id for c in train}
    dev_ids = {c.id for c in dev}
    test_ids = {c.id for c in test}
    assert train_ids.isdisjoint(dev_ids)
    assert train_ids.isdisjoint(test_ids)
    assert dev_ids.isdisjoint(test_ids)


def test_split_covers_all():
    cuts = _make_cuts(200, duration=10.0)
    train, dev, test = split_cuts(cuts, dev_hours=0.1, test_hours=0.1)
    assert len(train) + len(dev) + len(test) == 200


def test_split_dev_meets_budget():
    cuts = _make_cuts(500, duration=10.0)  # 5000s total
    train, dev, test = split_cuts(cuts, dev_hours=1.0, test_hours=1.0)
    dev_dur = sum(c.duration for c in dev)
    assert dev_dur >= 3600.0
```

- [ ] **Step 2: Run tests — expect ImportError**

```bash
pytest tests/test_transform.py -v
```

Expected: `ImportError: cannot import name 'is_valid' from 'canary_sk.transform'`

- [ ] **Step 3: Create `src/canary_sk/transform.py`**

```python
from collections import defaultdict

import numpy as np

from canary_sk.normalize import normalize_text


def is_valid(duration: float, text: str) -> bool:
    if duration < 1.0 or duration > 40.0:
        return False
    if len(text) < 5:
        return False
    cps = len(text) / duration
    if cps < 3.0 or cps > 30.0:
        return False
    return True


def stratified_indices(rows_meta: list, target_hours: float, seed: int = 42) -> set:
    """Select indices proportionally across snapshot years so total duration ≈ target_hours."""
    rng = np.random.default_rng(seed)
    by_year: dict = defaultdict(list)
    for i, meta in enumerate(rows_meta):
        year = str(meta["snapshot"])[:4]
        by_year[year].append((i, meta["duration"]))

    total_seconds = sum(d for ys in by_year.values() for _, d in ys)
    if total_seconds == 0:
        return set()

    fraction = min(1.0, (target_hours * 3600) / total_seconds)
    selected = []
    for items in by_year.values():
        rng.shuffle(items)
        budget = sum(d for _, d in items) * fraction
        running = 0.0
        for idx, dur in items:
            if running >= budget:
                break
            selected.append(idx)
            running += dur
    return set(selected)


def split_cuts(cuts: list, dev_hours: float = 3.0, test_hours: float = 3.0, seed: int = 42):
    """Split cuts into train/dev/test by session hash to prevent data leakage."""
    rng = np.random.default_rng(seed)
    by_session: dict = defaultdict(list)
    for c in cuts:
        by_session[hash(c.id) % 1000].append(c)

    sessions = list(by_session.keys())
    rng.shuffle(sessions)

    dev, test, train = [], [], []
    dev_budget = dev_hours * 3600
    test_budget = test_hours * 3600

    for s in sessions:
        if sum(c.duration for c in dev) < dev_budget:
            dev.extend(by_session[s])
        elif sum(c.duration for c in test) < test_budget:
            test.extend(by_session[s])
        else:
            train.extend(by_session[s])

    return train, dev, test


def row_to_cut(row: dict):
    """Convert a SloPalSpeech HF dataset row to a Lhotse MonoCut. Returns None if invalid."""
    # Lhotse imported lazily — not available in the Mac dev environment
    from lhotse import MonoCut, Recording, SupervisionSegment

    text = normalize_text(row["text"])
    duration = float(row["duration"])
    if not is_valid(duration, text):
        return None

    audio_array = np.asarray(row["audio"]["array"], dtype=np.float32)
    sr = int(row["audio"]["sampling_rate"])
    cut_id = f"slopal_{row['id']}"

    recording = Recording.from_array(audio_array, sampling_rate=sr, recording_id=cut_id)
    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=duration,
        text=text,
        language="sk",
    )
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
    )
    cut.custom = {"source_lang": "sk", "target_lang": "sk", "task": "asr", "pnc": "yes"}
    return cut
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
pytest tests/test_transform.py -v
```

Expected:
```
tests/test_transform.py::test_is_valid_too_short PASSED
tests/test_transform.py::test_is_valid_boundary_min PASSED
tests/test_transform.py::test_is_valid_too_long PASSED
tests/test_transform.py::test_is_valid_boundary_max PASSED
tests/test_transform.py::test_is_valid_cps_too_low PASSED
tests/test_transform.py::test_is_valid_cps_too_high PASSED
tests/test_transform.py::test_is_valid_text_too_short PASSED
tests/test_transform.py::test_stratified_zero_hours PASSED
tests/test_transform.py::test_stratified_large_target_selects_all PASSED
tests/test_transform.py::test_stratified_empty_input PASSED
tests/test_transform.py::test_stratified_proportional PASSED
tests/test_transform.py::test_split_no_overlap PASSED
tests/test_transform.py::test_split_covers_all PASSED
tests/test_transform.py::test_split_dev_meets_budget PASSED
14 passed
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: 22 passed (8 normalize + 14 transform)

- [ ] **Step 6: Commit**

```bash
git add tests/test_transform.py src/canary_sk/transform.py
git commit -m "feat: add transform module with TDD tests"
```

---

## Task 4: benchmark.py library module

**Files:**
- Create: `src/canary_sk/benchmark.py`

> No local unit tests — `evaluate_dataset` requires a NeMo model and GPU. This module is only run on RunPod. The lhotse imports are avoided (not needed here); NeMo is imported lazily in the CLI script.

- [ ] **Step 1: Create `src/canary_sk/benchmark.py`**

```python
import tempfile
import time
from pathlib import Path

import jiwer
import soundfile as sf
import torch

from canary_sk.normalize import normalize_for_wer


def evaluate_dataset(
    model,
    dataset,
    name: str,
    audio_key: str,
    text_key: str,
    max_samples: int | None = None,
    batch_size: int = 8,
) -> dict:
    print(f"\n=== Evaluating {name} ===")
    total = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"    samples: {total}")

    refs, hyps, audio_paths = [], [], []
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        print("    [1/2] Writing audio files...")
        for i, sample in enumerate(dataset):
            if i >= total:
                break
            audio = sample[audio_key]
            ap = Path(tmpdir) / f"{i:06d}.wav"
            sf.write(ap, audio["array"], audio["sampling_rate"])
            audio_paths.append(str(ap))
            refs.append(normalize_for_wer(sample[text_key]))

        print(f"    [2/2] Transcribing in batches of {batch_size}...")
        for start in range(0, len(audio_paths), batch_size):
            batch = audio_paths[start : start + batch_size]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.transcribe(
                    batch,
                    source_lang="sk",
                    target_lang="sk",
                    task="asr",
                    pnc="yes",
                    batch_size=batch_size,
                    verbose=False,
                )
            for o in out:
                txt = o.text if hasattr(o, "text") else (o if isinstance(o, str) else str(o))
                hyps.append(normalize_for_wer(txt))

            done = min(start + batch_size, len(audio_paths))
            if done % (batch_size * 5) == 0 or done == len(audio_paths):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (len(audio_paths) - done) / max(rate, 1e-3)
                print(f"      {done}/{len(audio_paths)} ({rate:.1f} samples/s, ETA {eta:.0f}s)")

    wer = jiwer.wer(refs, hyps)
    cer = jiwer.cer(refs, hyps)
    print(f"\n    >>> {name}")
    print(f"        WER: {wer * 100:.2f}%")
    print(f"        CER: {cer * 100:.2f}%")

    print("\n    Sample predictions:")
    for i in range(min(3, len(refs))):
        print(f"      REF: {refs[i][:80]}")
        print(f"      HYP: {hyps[i][:80]}")
        print()

    return {"name": name, "wer": wer, "cer": cer, "n": len(refs)}
```

- [ ] **Step 2: Verify all existing tests still pass**

```bash
pytest tests/ -v
```

Expected: 22 passed (no regressions)

- [ ] **Step 3: Commit**

```bash
git add src/canary_sk/benchmark.py
git commit -m "feat: add benchmark library module"
```

---

## Task 5: CLI scripts

**Files:**
- Create: `scripts/transform_slopal.py`
- Create: `scripts/benchmark.py`

- [ ] **Step 1: Create `scripts/transform_slopal.py`**

```python
"""
SloPalSpeech → Lhotse Shar transformation for Canary-1B-v2 fine-tuning.

Usage (on RunPod):
    python scripts/transform_slopal.py \
        --output-dir /workspace/slopal_lhotse \
        --target-hours 500 \
        --num-jobs 16
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from lhotse.shar import SharWriter

from canary_sk.transform import row_to_cut, split_cuts, stratified_indices


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target-hours", type=float, default=500.0,
                   help="Target subset in hours. -1 = full dataset.")
    p.add_argument("--shard-size", type=int, default=2000,
                   help="Cuts per shard.")
    p.add_argument("--num-jobs", type=int, default=8)
    p.add_argument("--hf-cache", default=None)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(">>> Loading SloPalSpeech metadata...")
    ds = load_dataset(
        "NaiveNeuron/SloPalSpeech",
        split="train",
        cache_dir=args.hf_cache,
    )
    print(f"    {len(ds):,} segments loaded.")

    if args.target_hours > 0:
        print(f">>> Stratified sampling for ~{args.target_hours} hours...")
        meta = [{"snapshot": ds[i]["snapshot"], "duration": ds[i]["duration"]}
                for i in range(len(ds))]
        keep = stratified_indices(meta, args.target_hours)
        print(f"    Selected {len(keep):,} / {len(ds):,} segments.")
    else:
        keep = None

    print(">>> Building cuts...")
    cuts = []
    skipped = 0
    for i, row in enumerate(ds):
        if keep is not None and i not in keep:
            continue
        cut = row_to_cut(row)
        if cut is None:
            skipped += 1
            continue
        cuts.append(cut)
        if len(cuts) % 5000 == 0:
            print(f"    {len(cuts):,} cuts built (skipped {skipped})")

    print(f">>> {len(cuts):,} valid cuts (skipped {skipped} invalid).")

    print(">>> Splitting train/dev/test...")
    train, dev, test = split_cuts(cuts, dev_hours=3.0, test_hours=3.0)
    print(f"    train={len(train)} ({sum(c.duration for c in train)/3600:.1f}h)")
    print(f"    dev=  {len(dev)}   ({sum(c.duration for c in dev)/3600:.1f}h)")
    print(f"    test= {len(test)}  ({sum(c.duration for c in test)/3600:.1f}h)")

    for split_name, split_cuts_list in [("train", train), ("dev", dev), ("test", test)]:
        split_dir = out / split_name
        split_dir.mkdir(exist_ok=True)
        print(f">>> Writing {split_name} shards to {split_dir}...")
        with SharWriter(
            str(split_dir),
            shard_size=args.shard_size,
            fields={"recording": "flac"},
        ) as writer:
            for cut in split_cuts_list:
                writer.write(cut)
        print(f"    done: {split_name}")

    print(">>> ALL DONE")
    print(f"    Output: {out.resolve()}")
    print("Use in NeMo config:")
    print(f"  train_ds.shar_path: {out.resolve()}/train")
    print(f"  validation_ds.shar_path: {out.resolve()}/dev")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create `scripts/benchmark.py`**

```python
"""
Benchmark Canary (pretrained or fine-tuned) on CommonVoice 21 SK + FLEURS SK.

Usage (on RunPod):
    # Pretrained baseline:
    python scripts/benchmark.py --pretrained nvidia/canary-1b-v2

    # Fine-tuned model:
    python scripts/benchmark.py --model /workspace/exp/.../checkpoints/best.nemo

    # Quick test (100 samples):
    python scripts/benchmark.py --pretrained nvidia/canary-1b-v2 --max-samples 100
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from canary_sk.benchmark import evaluate_dataset


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Path to .nemo file (fine-tuned)")
    g.add_argument("--pretrained", help="HuggingFace model name, e.g. nvidia/canary-1b-v2")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit samples per dataset (quick test)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--skip-cv", action="store_true")
    p.add_argument("--skip-fleurs", action="store_true")
    args = p.parse_args()

    # NeMo imported here — only available on RunPod with nemo_toolkit installed
    from nemo.collections.asr.models import EncDecMultiTaskModel

    print("Loading model...")
    if args.model:
        model = EncDecMultiTaskModel.restore_from(args.model)
        model_name = Path(args.model).stem
    else:
        model = EncDecMultiTaskModel.from_pretrained(args.pretrained)
        model_name = args.pretrained

    model = model.cuda().eval()
    print(f"Model: {model_name}")

    results = []

    if not args.skip_cv:
        print("\nLoading CommonVoice 21 SK test split...")
        cv = load_dataset(
            "mozilla-foundation/common_voice_21_0",
            "sk",
            split="test",
            trust_remote_code=True,
        )
        results.append(evaluate_dataset(
            model, cv, "CommonVoice 21 SK", "audio", "sentence",
            max_samples=args.max_samples, batch_size=args.batch_size,
        ))

    if not args.skip_fleurs:
        print("\nLoading FLEURS SK test split...")
        fleurs = load_dataset("google/fleurs", "sk_sk", split="test")
        text_key = "transcription" if "transcription" in fleurs.features else "raw_transcription"
        results.append(evaluate_dataset(
            model, fleurs, "FLEURS SK", "audio", text_key,
            max_samples=args.max_samples, batch_size=args.batch_size,
        ))

    print("\n" + "=" * 50)
    print(f"  FINAL RESULTS — {model_name}")
    print("=" * 50)
    for r in results:
        print(f"  {r['name']:25} WER={r['wer']*100:6.2f}%  CER={r['cer']*100:6.2f}%  (n={r['n']})")
    print("=" * 50)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify all tests still pass**

```bash
pytest tests/ -v
```

Expected: 22 passed

- [ ] **Step 4: Commit**

```bash
git add scripts/transform_slopal.py scripts/benchmark.py
git commit -m "feat: add CLI script wrappers"
```

---

## Task 6: Remove old flat files

**Files:**
- Delete: `transform_slopal_to_lhotse.py`
- Delete: `benchmark.py` (root-level)

- [ ] **Step 1: Delete the old files**

```bash
git rm transform_slopal_to_lhotse.py benchmark.py
```

- [ ] **Step 2: Run full test suite — verify nothing broke**

```bash
pytest tests/ -v
```

Expected: 22 passed

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove old flat scripts (replaced by src/canary_sk/ + scripts/)"
```

---

## Task 7: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a "Local development (Mac)" section near the top of README.md**

Add this block right after the `## Goal` section and before `## Key constraints`:

```markdown
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

These tests cover text normalisation, audio filtering rules, stratified sampling, and train/dev/test split logic. They run in <5s with no downloads.
```

- [ ] **Step 2: Update the "Files I want in this repo" section**

Replace the `transform_slopal_to_lhotse.py` and `benchmark.py` bullet points with:

```markdown
- `src/canary_sk/normalize.py` — `normalize_text()` (training, keeps PnC) and `normalize_for_wer()` (benchmark evaluation, lowercase + strip punct).
- `src/canary_sk/transform.py` — all SloPalSpeech conversion logic: `row_to_cut`, `is_valid`, `stratified_indices`, `split_cuts`. Lhotse imported lazily inside `row_to_cut` so the module is importable on Mac.
- `src/canary_sk/benchmark.py` — `evaluate_dataset()` used by the benchmark CLI. Only runs on RunPod (requires torch, soundfile, jiwer).
- `scripts/transform_slopal.py` — thin CLI wrapper: argparse + calls `canary_sk.transform`.
- `scripts/benchmark.py` — thin CLI wrapper: argparse + calls `canary_sk.benchmark`. Imports NeMo lazily.
- `tests/` — unit tests for all pure-Python logic. Run with `pytest tests/`.
- `pyproject.toml` — makes `canary_sk` importable as a package; configures pytest.
- `requirements.txt` — RunPod full stack: `nemo_toolkit[asr]>=2.0`, `lhotse>=1.30`, `jiwer>=3.0`, `datasets>=2.20`, `soundfile`.
- `requirements-dev.txt` — Mac lightweight: `numpy>=1.26`, `pytest>=8.0`.
```

- [ ] **Step 3: Update the three RunPod commands in README to use the new script paths**

Find any reference to the old command form and update to:

```bash
# 1. Transform dataset
python scripts/transform_slopal.py --output-dir /workspace/slopal_lhotse --target-hours 500

# 2. Fine-tune
python -m nemo.collections.asr.scripts.speech_to_text_finetune \
    --config-path=/workspace --config-name=finetune

# 3. Benchmark
python scripts/benchmark.py --pretrained nvidia/canary-1b-v2
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README with uv setup and new script paths"
```

---

## Final verification

- [ ] **Run the full test suite one last time**

```bash
pytest tests/ -v
```

Expected: 22 passed, 0 failed, 0 errors

- [ ] **Verify project structure matches the design**

```bash
find . -not -path './.venv/*' -not -path './.git/*' -not -path './__pycache__/*' | sort
```

Expected output includes:
```
./README.md
./docs/superpowers/plans/2026-05-19-project-structure-refactor.md
./docs/superpowers/specs/2026-05-19-project-structure-design.md
./finetune.yaml
./pyproject.toml
./requirements-dev.txt
./requirements.txt
./scripts/benchmark.py
./scripts/transform_slopal.py
./src/canary_sk/__init__.py
./src/canary_sk/benchmark.py
./src/canary_sk/normalize.py
./src/canary_sk/transform.py
./tests/conftest.py
./tests/test_normalize.py
./tests/test_transform.py
```

- [ ] **Verify old files are gone**

```bash
ls transform_slopal_to_lhotse.py benchmark.py 2>&1
```

Expected: `No such file or directory` for both
