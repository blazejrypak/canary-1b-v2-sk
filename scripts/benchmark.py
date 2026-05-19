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
