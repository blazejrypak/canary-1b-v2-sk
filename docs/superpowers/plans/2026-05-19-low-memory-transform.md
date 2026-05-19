# Low-Memory Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `transform_slopal.py` so peak RAM stays under 16 GB by streaming audio cuts directly to disk instead of accumulating them all in memory before splitting/writing.

**Architecture:** Add `assign_splits()` to `transform.py` that pre-computes `{idx → split_name}` using only IDs and durations (no audio). Extend Pass 1 to also collect row IDs. Refactor Pass 2 to open three `SharWriter` contexts simultaneously and route each cut immediately to its destination split — no list accumulation, no `split_cuts()` call. Fix `row_to_cut` to derive duration from the actual audio array (`len(array)/sr`) so cut, supervision, and recording durations are always consistent.

**Tech Stack:** Python 3.11, Lhotse ≥ 1.30, NumPy, HuggingFace `datasets` (streaming mode), pytest

---

## File Map

| File | Change |
|---|---|
| `src/canary_sk/transform.py` | Add `assign_splits()`, fix `row_to_cut` duration |
| `tests/test_transform.py` | Add tests for `assign_splits`, add `row_to_cut` duration tests |
| `scripts/transform_slopal.py` | Collect IDs in Pass 1, stream-write in Pass 2 |
| `docs/runpod-guide.md` | Remove phantom `--num-jobs` arg, fix expected output numbers |

---

## Task 1: `assign_splits` — deterministic per-cut split assignment

**Files:**
- Modify: `src/canary_sk/transform.py`
- Test: `tests/test_transform.py`

`assign_splits` takes `{idx: (cut_id, duration)}` (available after Pass 1, no audio needed) and returns `{idx: split_name}`. It sorts entries by MD5 hex of `cut_id` for a stable ordering, then fills dev and test budgets before assigning the rest to train. Purely deterministic — no RNG, no seed parameter.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transform.py`:

```python
# ---- assign_splits ----

from canary_sk.transform import assign_splits


def _make_kept(n: int, duration: float = 10.0, prefix: str = "cut") -> dict:
    """Helper: {idx: (cut_id, duration)} for n cuts."""
    return {i: (f"slopal_{prefix}_{i:05d}", duration) for i in range(n)}


def test_assign_splits_empty():
    assert assign_splits({}) == {}


def test_assign_splits_all_indices_present():
    kept = _make_kept(50)
    result = assign_splits(kept, dev_hours=0.05, test_hours=0.05)
    assert set(result.keys()) == set(kept.keys())


def test_assign_splits_valid_split_names():
    kept = _make_kept(50)
    result = assign_splits(kept, dev_hours=0.05, test_hours=0.05)
    assert set(result.values()) <= {"train", "dev", "test"}


def test_assign_splits_no_overlap():
    kept = _make_kept(200, duration=10.0)
    result = assign_splits(kept, dev_hours=0.1, test_hours=0.1)
    by_split = {"train": set(), "dev": set(), "test": set()}
    for idx, split in result.items():
        by_split[split].add(idx)
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["dev"].isdisjoint(by_split["test"])


def test_assign_splits_dev_meets_budget():
    kept = _make_kept(500, duration=10.0)  # 5000s total
    result = assign_splits(kept, dev_hours=1.0, test_hours=1.0)
    dev_dur = sum(kept[i][1] for i, s in result.items() if s == "dev")
    assert dev_dur >= 3600.0


def test_assign_splits_test_meets_budget():
    kept = _make_kept(500, duration=10.0)
    result = assign_splits(kept, dev_hours=1.0, test_hours=1.0)
    test_dur = sum(kept[i][1] for i, s in result.items() if s == "test")
    assert test_dur >= 3600.0


def test_assign_splits_deterministic():
    kept = _make_kept(100)
    r1 = assign_splits(kept, dev_hours=0.1, test_hours=0.1)
    r2 = assign_splits(kept, dev_hours=0.1, test_hours=0.1)
    assert r1 == r2


def test_assign_splits_large_budget_puts_all_in_dev():
    kept = _make_kept(10, duration=5.0)  # 50s total
    result = assign_splits(kept, dev_hours=999.0, test_hours=0.0)
    assert all(s == "dev" for s in result.values())
```

- [ ] **Step 2: Run to confirm they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_transform.py -k "assign_splits" -v
```

Expected: `ImportError` or `FAILED` on `assign_splits` not found — confirm failure before implementing.

- [ ] **Step 3: Implement `assign_splits` in `transform.py`**

Add after the `split_cuts` function in `src/canary_sk/transform.py`:

```python
def assign_splits(
    kept: dict,
    dev_hours: float = 3.0,
    test_hours: float = 3.0,
) -> dict:
    """
    Pre-compute {idx: split_name} using only cut IDs and durations.

    Sorts entries by MD5 hex of cut_id for a stable, reproducible ordering,
    then fills dev and test hour budgets before assigning the rest to train.
    Requires no audio data — safe to call after Pass 1.
    """
    if not kept:
        return {}
    dev_budget = dev_hours * 3600
    test_budget = test_hours * 3600
    ordered = sorted(kept.items(), key=lambda kv: hashlib.md5(kv[1][0].encode()).hexdigest())
    result: dict = {}
    dev_s = test_s = 0.0
    for idx, (cut_id, dur) in ordered:
        if dev_s < dev_budget:
            result[idx] = "dev"
            dev_s += dur
        elif test_s < test_budget:
            result[idx] = "test"
            test_s += dur
        else:
            result[idx] = "train"
    return result
```

Note: `hashlib` is already imported at the top of `transform.py`.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_transform.py -k "assign_splits" -v
```

Expected: all 8 `assign_splits` tests `PASSED`.

- [ ] **Step 5: Run the full suite to confirm no regressions**

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: all 30 tests `PASSED` (22 original + 8 new).

- [ ] **Step 6: Commit**

```bash
git add src/canary_sk/transform.py tests/test_transform.py
git commit -m "feat: add assign_splits for memory-safe pre-computed split assignment"
```

---

## Task 2: Fix `row_to_cut` to use actual audio duration

**Files:**
- Modify: `src/canary_sk/transform.py`
- Test: `tests/test_transform.py`

Currently `row_to_cut` filters via `row["duration"]` (metadata) but builds the `Recording` from the array — meaning recording duration and supervision/cut duration can silently diverge. Fix: decode the array first, derive duration from `len(array)/sr`, use that single value everywhere.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_transform.py`. These tests mock `lhotse` so they run in the local venv where lhotse is not installed:

```python
# ---- row_to_cut duration ----

import sys
from unittest.mock import MagicMock, patch

from canary_sk.transform import row_to_cut


def _mock_lhotse():
    mock = MagicMock()
    mock.MonoCut.return_value = MagicMock()
    mock.Recording.from_array.return_value = MagicMock()
    mock.SupervisionSegment.return_value = MagicMock()
    return mock


def test_row_to_cut_uses_actual_duration_not_metadata():
    """Metadata says 0.5s (invalid), but actual array is 10.0s (valid) → cut produced."""
    with patch.dict(sys.modules, {"lhotse": _mock_lhotse()}):
        row = {
            "id": "test_001",
            "text": "Testovacia veta pre parlamentný prejav",   # 38 chars, ~3.8 cps at 10s
            "duration": 0.5,                                   # metadata: too short
            "audio": {"array": [0.0] * 160_000, "sampling_rate": 16_000},  # actual: 10.0s
        }
        result = row_to_cut(row)
        assert result is not None


def test_row_to_cut_rejects_on_actual_duration():
    """Metadata says 10.0s (valid), but actual array is 0.5s (invalid) → None."""
    with patch.dict(sys.modules, {"lhotse": _mock_lhotse()}):
        row = {
            "id": "test_002",
            "text": "Testovacia veta pre parlamentný prejav",
            "duration": 10.0,                                  # metadata: fine
            "audio": {"array": [0.0] * 8_000, "sampling_rate": 16_000},    # actual: 0.5s
        }
        result = row_to_cut(row)
        assert result is None
```

- [ ] **Step 2: Run to confirm they fail**

```bash
source .venv/bin/activate && python -m pytest tests/test_transform.py -k "actual_duration" -v
```

Expected: both tests `FAILED` — `row_to_cut` currently reads `row["duration"]` (metadata) so the assertions are inverted.

- [ ] **Step 3: Fix `row_to_cut` in `transform.py`**

Replace the existing `row_to_cut` function body:

Old (lines 77–109 in `src/canary_sk/transform.py`):
```python
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

New:
```python
def row_to_cut(row: dict):
    """Convert a SloPalSpeech HF dataset row to a Lhotse MonoCut. Returns None if invalid."""
    # Lhotse imported lazily — not available in the Mac dev environment
    from lhotse import MonoCut, Recording, SupervisionSegment

    audio_array = np.asarray(row["audio"]["array"], dtype=np.float32)
    sr = int(row["audio"]["sampling_rate"])
    duration = len(audio_array) / sr  # actual duration from array, not metadata field

    text = normalize_text(row["text"])
    if not is_valid(duration, text):
        return None

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

- [ ] **Step 4: Run tests to confirm they pass**

```bash
source .venv/bin/activate && python -m pytest tests/test_transform.py -k "actual_duration" -v
```

Expected: both `PASSED`.

- [ ] **Step 5: Run the full suite**

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: all 32 tests `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add src/canary_sk/transform.py tests/test_transform.py
git commit -m "fix: row_to_cut uses actual array duration, not metadata field"
```

---

## Task 3: Refactor `transform_slopal.py` to stream-write cuts

**Files:**
- Modify: `scripts/transform_slopal.py`

Replace the current pattern (collect all cuts → split → write) with: pre-compute split assignment in Pass 1, then write each cut immediately to its SharWriter in Pass 2. Peak RAM drops from ~23 GB to ~2–4 GB.

**Key changes:**
1. Pass 1: also collect `id_list` (row IDs, no audio — cheap ~7 MB for 350k rows)
2. After Pass 1: compute `kept_ids` dict and call `assign_splits`
3. Pass 2: open three `SharWriter` contexts simultaneously; `write(cut)` immediately after `row_to_cut`

- [ ] **Step 1: Replace `scripts/transform_slopal.py` with the refactored version**

Full replacement — the entire file:

```python
"""
SloPalSpeech → Lhotse Shar transformation for Canary-1B-v2 fine-tuning.

Two-pass streaming — no local 60 GB parquet cache needed:
  Pass 1: streams metadata columns only (snapshot, duration, id, text).
          Parquet column projection skips the audio binary entirely.
          Also collects row IDs needed for deterministic split assignment.
  Pass 2: streams all columns, decodes audio only for the selected rows
          (~3.5% of rows for --target-hours 100 out of 2806 h total).
          Cuts are written immediately — no in-memory accumulation.
          Peak RAM: ~2-4 GB regardless of target hours.

Works both locally (Mac) and on RunPod. Requires:
    pip install datasets lhotse soundfile numpy

Usage:
    python scripts/transform_slopal.py --output-dir ./slopal_lhotse
    python scripts/transform_slopal.py --output-dir /workspace/slopal_lhotse --target-hours 100
    python scripts/transform_slopal.py --output-dir /workspace/slopal_lhotse --target-hours -1  # full dataset
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from lhotse.shar import SharWriter

from canary_sk.transform import assign_splits, row_to_cut, stratified_indices


def _stream_dataset(hf_cache: str | None, columns: list[str] | None = None):
    ds = load_dataset(
        "NaiveNeuron/SloPalSpeech",
        split="train",
        streaming=True,
        cache_dir=hf_cache,
    )
    if columns:
        ds = ds.select_columns(columns)
    return ds


def main():
    p = argparse.ArgumentParser(
        description="Convert SloPalSpeech to Lhotse Shar shards via streaming."
    )
    p.add_argument("--output-dir", required=True,
                   help="Directory where train/dev/test shard folders are written.")
    p.add_argument("--target-hours", type=float, default=100.0,
                   help="Target subset size in hours. -1 = full dataset (2806 h).")
    p.add_argument("--shard-size", type=int, default=2000,
                   help="Cuts per Shar shard file.")
    p.add_argument("--hf-cache", default=None,
                   help="HuggingFace datasets cache directory (optional).")
    p.add_argument("--dev-hours", type=float, default=3.0,
                   help="Hours to allocate to the dev split.")
    p.add_argument("--test-hours", type=float, default=3.0,
                   help="Hours to allocate to the test split.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: stream metadata only — parquet skips the audio column.
    # Collect rows_meta for stratified sampling and id_list for split
    # assignment. Both are small: ~21 MB for 350k rows.
    # ------------------------------------------------------------------
    print(">>> Pass 1: streaming metadata (audio column skipped)...")
    rows_meta: list[dict] = []
    id_list: list[str] = []
    for i, row in enumerate(_stream_dataset(args.hf_cache, columns=["id", "snapshot", "duration", "text"])):
        rows_meta.append({"snapshot": row["snapshot"], "duration": row["duration"]})
        id_list.append(row["id"])
        if (i + 1) % 50_000 == 0:
            hrs = sum(r["duration"] for r in rows_meta) / 3600
            print(f"    {i + 1:,} rows scanned ({hrs:.0f} h accumulated)...")

    total = len(rows_meta)
    total_hrs = sum(r["duration"] for r in rows_meta) / 3600
    print(f"    {total:,} segments found ({total_hrs:.0f} h total).")

    if args.target_hours > 0:
        print(f">>> Stratified sampling for ~{args.target_hours} h...")
        keep = stratified_indices(rows_meta, args.target_hours)
        print(f"    Selected {len(keep):,} / {total:,} segments ({len(keep) / total * 100:.1f}%).")
    else:
        keep = None
        print(">>> Using full dataset (no sampling).")

    # ------------------------------------------------------------------
    # Pre-compute split assignment from IDs and durations only (no audio).
    # assign_splits sorts by MD5 hash of cut_id for a stable ordering, then
    # fills dev and test budgets before assigning the rest to train.
    # ------------------------------------------------------------------
    print(">>> Computing train/dev/test assignment...")
    kept_ids: dict = {}
    indices = keep if keep is not None else range(total)
    for idx in indices:
        cut_id = f"slopal_{id_list[idx]}"
        kept_ids[idx] = (cut_id, rows_meta[idx]["duration"])

    split_assignment = assign_splits(kept_ids, dev_hours=args.dev_hours, test_hours=args.test_hours)

    split_counts = {"train": 0, "dev": 0, "test": 0}
    split_hours = {"train": 0.0, "dev": 0.0, "test": 0.0}
    for idx, split in split_assignment.items():
        split_counts[split] += 1
        split_hours[split] += kept_ids[idx][1]
    print(f"    train={split_counts['train']:,} ({split_hours['train'] / 3600:.1f} h)  "
          f"dev={split_counts['dev']:,} ({split_hours['dev'] / 3600:.1f} h)  "
          f"test={split_counts['test']:,} ({split_hours['test'] / 3600:.1f} h)")

    # ------------------------------------------------------------------
    # Pass 2: stream all columns, decode audio only for selected rows.
    # Write each cut immediately — no in-memory accumulation.
    # Peak RAM = one audio array at a time (~3 MB max) + writer buffers.
    # ------------------------------------------------------------------
    print(">>> Pass 2: streaming audio and writing shards...")
    for name in ("train", "dev", "test"):
        (out / name).mkdir(exist_ok=True)

    skipped = 0
    written = {"train": 0, "dev": 0, "test": 0}
    written_hrs = {"train": 0.0, "dev": 0.0, "test": 0.0}

    with (
        SharWriter(str(out / "train"), shard_size=args.shard_size, fields={"recording": "flac"}) as tw,
        SharWriter(str(out / "dev"),   shard_size=args.shard_size, fields={"recording": "flac"}) as dw,
        SharWriter(str(out / "test"),  shard_size=args.shard_size, fields={"recording": "flac"}) as testw,
    ):
        writers = {"train": tw, "dev": dw, "test": testw}
        for i, row in enumerate(_stream_dataset(args.hf_cache)):
            if i not in split_assignment:
                continue
            cut = row_to_cut(row)
            if cut is None:
                skipped += 1
                continue
            split = split_assignment[i]
            writers[split].write(cut)
            written[split] += 1
            written_hrs[split] += cut.duration
            total_written = sum(written.values())
            if total_written % 1000 == 0:
                pct = i / total * 100
                print(f"    {total_written:,} cuts written | skipped {skipped} | "
                      f"scanned {i + 1:,} rows ({pct:.0f}%)")

    total_written = sum(written.values())
    print(f">>> {total_written:,} valid cuts written ({skipped} filtered out).")
    print(f"    train={written['train']:,} ({written_hrs['train'] / 3600:.1f} h)")
    print(f"    dev=  {written['dev']:,} ({written_hrs['dev'] / 3600:.1f} h)")
    print(f"    test= {written['test']:,} ({written_hrs['test'] / 3600:.1f} h)")

    print("\n>>> ALL DONE")
    print(f"    Output: {out.resolve()}")
    print(f"    Use in NeMo finetune.yaml:")
    print(f"      train_ds.shar_path: {out.resolve()}/train")
    print(f"      validation_ds.shar_path: {out.resolve()}/dev")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script imports correctly**

```bash
source .venv/bin/activate && python -c "
import ast, pathlib
src = pathlib.Path('scripts/transform_slopal.py').read_text()
ast.parse(src)
print('syntax OK')
"
```

Expected: `syntax OK`

- [ ] **Step 3: Verify `assign_splits` is imported in the script**

```bash
grep "assign_splits" scripts/transform_slopal.py
```

Expected: two lines — the import and the call site.

- [ ] **Step 4: Run full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

Expected: all 32 tests `PASSED` (the script itself is not covered by unit tests since it requires HF + lhotse, but the library functions it calls are fully tested).

- [ ] **Step 5: Commit**

```bash
git add scripts/transform_slopal.py
git commit -m "refactor: stream-write cuts to avoid 23 GB peak RAM in transform_slopal"
```

---

## Task 4: Fix `runpod-guide.md` documentation

**Files:**
- Modify: `docs/runpod-guide.md`

Two documentation bugs: (1) "Known Issues" mentions a `--num-jobs` argument that doesn't exist in the current script; (2) the expected output under Step 1 shows 500h numbers inconsistent with the 100h default.

- [ ] **Step 1: Remove the phantom `--num-jobs` entry**

In `docs/runpod-guide.md`, find and remove the paragraph under "Known Issues and Warnings" that reads:

```
**`--num-jobs` argument in transform_slopal.py is unused.**
The dataset transformation runs single-threaded. The argument is parsed but never applied. On A100 this still completes in ~2h. If it's too slow, the fix would be to parallelize `row_to_cut` calls with `multiprocessing.Pool`.
```

- [ ] **Step 2: Fix the expected output block**

Find the "Expected output:" block under Step 1 (Transform SloPalSpeech) and replace it:

Old:
```
**Expected output:**
\`\`\`
>>> Loading SloPalSpeech metadata...
    ~350,000 segments loaded.
>>> Stratified sampling for ~500 hours...
    Selected ~70,000 / ~350,000 segments.
>>> Building cuts...
>>> Splitting train/dev/test...
    train=~69,000 (494.0h)
    dev=  ~430   (3.0h)
    test= ~430   (3.0h)
>>> Writing train shards...
>>> ALL DONE
    Output: /workspace/slopal_lhotse
\`\`\`
```

New:
```
**Expected output:**
\`\`\`
>>> Pass 1: streaming metadata (audio column skipped)...
    ~350,000 segments found (~2806 h total).
>>> Stratified sampling for ~100.0 h...
    Selected ~12,000 / ~350,000 segments (3.5%).
>>> Computing train/dev/test assignment...
    train=~11,100 (94.0 h)  dev=~430 (3.0 h)  test=~430 (3.0 h)
>>> Pass 2: streaming audio and writing shards...
>>> 11,500 valid cuts written (N filtered out).
>>> ALL DONE
    Output: /workspace/slopal_lhotse
\`\`\`
```

- [ ] **Step 3: Verify the guide no longer mentions `--num-jobs`**

```bash
grep -n "num-jobs\|num_jobs" docs/runpod-guide.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add docs/runpod-guide.md
git commit -m "docs: remove phantom --num-jobs arg, fix expected transform output to match 100h default"
```

---

## Self-Review

**Spec coverage:**
- Memory fix (23 GB → ~2 GB): covered by Task 3 (stream writes)
- `assign_splits` enabling stream writes: covered by Task 1
- Duration mismatch in `row_to_cut`: covered by Task 2
- `--num-jobs` phantom in docs: covered by Task 4
- Expected output numbers: covered by Task 4

**Placeholder scan:** No TBDs, no "add appropriate X", all code steps are complete.

**Type consistency:**
- `assign_splits(kept: dict, dev_hours, test_hours) -> dict` — called in Task 3 as `assign_splits(kept_ids, dev_hours=args.dev_hours, test_hours=args.test_hours)` ✓
- `row_to_cut(row: dict)` — unchanged signature ✓
- `stratified_indices` — unchanged ✓
- `SharWriter` used as context manager — same pattern as original script ✓
- `id_list` introduced in Pass 1, consumed after Pass 1 to build `kept_ids` ✓
