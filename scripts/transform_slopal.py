"""
SloPalSpeech → Lhotse Shar transformation for Canary-1B-v2 fine-tuning.

Two-pass streaming — no local 60 GB parquet cache needed:
  Pass 1: streams metadata columns only (snapshot, duration, id, text).
          Parquet column projection skips the audio binary entirely.
  Pass 2: streams all columns, decodes audio only for the selected rows
          (~3.5% of rows for --target-hours 100 out of 2806 h total).

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

from canary_sk.transform import row_to_cut, split_cuts, stratified_indices


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
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 1: stream metadata only — parquet skips the audio column
    # ------------------------------------------------------------------
    print(">>> Pass 1: streaming metadata (audio column skipped)...")
    rows_meta: list[dict] = []
    for i, row in enumerate(_stream_dataset(args.hf_cache, columns=["id", "snapshot", "duration", "text"])):
        rows_meta.append({"snapshot": row["snapshot"], "duration": row["duration"]})
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
    # Pass 2: stream all columns, decode audio only for selected rows
    # ------------------------------------------------------------------
    print(">>> Pass 2: streaming audio for selected segments...")
    cuts = []
    skipped = 0
    for i, row in enumerate(_stream_dataset(args.hf_cache)):
        if keep is not None and i not in keep:
            continue
        cut = row_to_cut(row)
        if cut is None:
            skipped += 1
            continue
        cuts.append(cut)
        if len(cuts) % 1000 == 0:
            hrs = sum(c.duration for c in cuts) / 3600
            pct = i / total * 100
            print(f"    {len(cuts):,} cuts ({hrs:.1f} h) | skipped {skipped} | scanned {i + 1:,} rows ({pct:.0f}%)")

    total_dur = sum(c.duration for c in cuts) / 3600
    print(f">>> {len(cuts):,} valid cuts ({total_dur:.1f} h, {skipped} filtered out).")

    # ------------------------------------------------------------------
    # Split and write Lhotse Shar shards
    # ------------------------------------------------------------------
    print(">>> Splitting train/dev/test...")
    train, dev, test = split_cuts(cuts, dev_hours=3.0, test_hours=3.0)
    print(f"    train={len(train):,}  ({sum(c.duration for c in train) / 3600:.1f} h)")
    print(f"    dev=  {len(dev):,}    ({sum(c.duration for c in dev) / 3600:.1f} h)")
    print(f"    test= {len(test):,}   ({sum(c.duration for c in test) / 3600:.1f} h)")

    for split_name, split_cuts_list in [("train", train), ("dev", dev), ("test", test)]:
        split_dir = out / split_name
        split_dir.mkdir(exist_ok=True)
        print(f">>> Writing {split_name} shards to {split_dir} ...")
        with SharWriter(str(split_dir), shard_size=args.shard_size, fields={"recording": "flac"}) as writer:
            for cut in split_cuts_list:
                writer.write(cut)
        print(f"    {split_name}: done.")

    print("\n>>> ALL DONE")
    print(f"    Output: {out.resolve()}")
    print(f"    Use in NeMo finetune.yaml:")
    print(f"      train_ds.shar_path: {out.resolve()}/train")
    print(f"      validation_ds.shar_path: {out.resolve()}/dev")


if __name__ == "__main__":
    main()
