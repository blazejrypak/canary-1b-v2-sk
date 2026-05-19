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
