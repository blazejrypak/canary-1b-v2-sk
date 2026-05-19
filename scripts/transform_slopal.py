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
import json
import os
from pathlib import Path

from datasets import Audio as HFAudio, load_dataset
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
    else:
        # Prevent datasets from auto-decoding audio (requires torchcodec in >=3.x).
        # row_to_cut decodes raw bytes via soundfile instead.
        ds = ds.cast_column("audio", HFAudio(decode=False))
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

    checkpoint = out / "pass1_checkpoint.json"

    # ------------------------------------------------------------------
    # Pass 1: stream metadata only — parquet skips the audio column.
    # Results are checkpointed to disk so a Pass 2 crash doesn't require
    # re-streaming 1 h of metadata on the next run.
    # ------------------------------------------------------------------
    if checkpoint.exists():
        print(f">>> Pass 1: loading checkpoint from {checkpoint} (skipping re-stream)...")
        ckpt = json.loads(checkpoint.read_text())
        split_assignment = {int(k): v for k, v in ckpt["split_assignment"].items()}
        total = ckpt["total"]
        print(f"    {total:,} segments, {len(split_assignment):,} selected.")
        split_counts = {"train": 0, "dev": 0, "test": 0}
        split_hours = {"train": 0.0, "dev": 0.0, "test": 0.0}
        for idx_str, split in ckpt["split_assignment"].items():
            split_counts[split] += 1
            split_hours[split] += ckpt["kept_durations"][idx_str]
        print(f"    train={split_counts['train']:,} ({split_hours['train'] / 3600:.1f} h)  "
              f"dev={split_counts['dev']:,} ({split_hours['dev'] / 3600:.1f} h)  "
              f"test={split_counts['test']:,} ({split_hours['test'] / 3600:.1f} h)")
    else:
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

        checkpoint.write_text(json.dumps({
            "total": total,
            "target_hours": args.target_hours,
            "split_assignment": {str(k): v for k, v in split_assignment.items()},
            "kept_durations": {str(idx): kept_ids[idx][1] for idx in split_assignment},
        }))
        print(f"    Checkpoint saved → {checkpoint}")

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
            tmp = (cut.custom or {}).pop("_tmp_path", None)
            if tmp:
                os.unlink(tmp)
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
