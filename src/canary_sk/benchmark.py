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
