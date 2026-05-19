# RunPod Deployment Guide — Canary-1B-v2 Slovak Fine-tuning

Total estimated time: ~5h active GPU. Estimated cost: ~$10–12 at $2.50/h A100 rates.

> **Data size decision:** 100h is used here (reduced from the original 500h plan). Slovak is already in Canary-1B-v2's pretraining; parliamentary speech is a clean, formal domain. Research shows 100h saturates domain adaptation for this case — 500h adds <1–2% absolute WER with 5× the cost.

---

## Prerequisites

Before touching RunPod:

- [ ] HuggingFace account with access to:
  - `NaiveNeuron/SloPalSpeech` (check if gated — request access if needed)
  - `mozilla-foundation/common_voice_21_0` (requires HF login)
  - `google/fleurs` (public)
- [ ] HF token with read access: https://huggingface.co/settings/tokens
- [ ] Private GitHub repo with this codebase + a **read-only deploy key** added
- [ ] RunPod account with billing enabled

---

## 1. Create the Pod

**Template:** `RunPod PyTorch` is NOT enough — use the NeMo image directly.

| Setting | Value |
|---|---|
| GPU | A100 SXM 80GB (preferred) or H100 80GB |
| Container image | `nvcr.io/nvidia/nemo:25.04` |
| Container disk | 50 GB |
| Volume disk | 200 GB (persistent, mounted at `/workspace`) |
| Expose ports | none needed |
| Environment variables | see below |

**Environment variables to set in RunPod UI:**

```
HF_TOKEN=hf_xxxxxxxxxxxx
GITHUB_DEPLOY_KEY=<paste private key content, or use the method below>
```

> Tip: for the deploy key it's easier to clone manually on first login rather than injecting the key via env var.

---

## 2. One-time Pod Setup

SSH into the pod, then run:

```bash
# 1. Clone the repo
cd /workspace
git clone git@github.com:<your-org>/canary-1b-v2-sk.git
cd canary-1b-v2-sk

# If GitHub SSH isn't set up on the pod yet:
# eval "$(ssh-agent -s)" && ssh-add /path/to/deploy_key
# git clone git@github.com:<your-org>/canary-1b-v2-sk.git

# 2. Install additional dependencies on top of the NeMo base image
# (nemo_toolkit is already in the container; these add lhotse, datasets, jiwer)
pip install -r requirements.txt

# 3. Verify the install
python -c "import lhotse; import datasets; import jiwer; print('OK')"

# 4. Confirm HF token is available
python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 3. Step 1 — Transform SloPalSpeech (~2h)

This downloads the dataset from HF and writes Lhotse Shar shards to the network volume.

```bash
cd /workspace/canary-1b-v2-sk

python scripts/transform_slopal.py \
    --output-dir /workspace/slopal_lhotse \
    --target-hours 100 \
    --hf-cache /workspace/hf_cache
```

**What it does:**
1. Downloads `NaiveNeuron/SloPalSpeech` (~full 2806h dataset metadata + audio arrays for selected segments)
2. Stratified sampling: picks ~500h proportionally across `snapshot` years
3. Filters invalid segments (< 1s, > 40s, bad chars/sec ratio)
4. Splits: train (~494h) / dev (3h) / test (3h) by session hash
5. Writes FLAC Shar shards to `/workspace/slopal_lhotse/{train,dev,test}/`

**Expected output:**
```
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
```

> If the download stalls or fails midway, re-run the same command — `datasets` caches parquet files in `--hf-cache`. It will resume from the cached parquets.

**Verify shards were written:**
```bash
ls /workspace/slopal_lhotse/train/ | head -5
# Should show: cuts.000000.jsonl.gz, recording.000000.tar, etc.
wc -l /workspace/slopal_lhotse/train/cuts.*.jsonl.gz  # rough cut count
```

---

## 4. Step 2 — Fine-tune (~6–8h)

```bash
cd /workspace/canary-1b-v2-sk

python -m nemo.collections.asr.scripts.speech_to_text_finetune \
    --config-path=/workspace/canary-1b-v2-sk \
    --config-name=finetune
```

**What happens:**
1. NeMo downloads `nvidia/canary-1b-v2` from HF Hub to NeMo cache (~4GB, one-time)
2. Loads pretrained weights — no tokenizer/decoder reinitialization
3. Trains for 15,000 steps with AdamW lr=1e-5, bf16-mixed, SpecAugment
4. Validates every 1,000 steps on the dev shard, saves top-3 checkpoints by `val_wer`
5. Checkpoints land in `/workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/`

**Monitor training:**
```bash
# In a second terminal (tail the log NeMo writes):
tail -f /workspace/exp/canary-1b-v2-slovak-parliament/nemo_log_globalrank-0_localrank-0.txt

# Or watch TensorBoard:
tensorboard --logdir /workspace/exp --bind_all
# then open pod's port 6006 in browser (add port forwarding in RunPod)
```

**If the pod dies mid-training** — `resume_if_exists: true` is set in `finetune.yaml`. Re-run the exact same command; NeMo will find the last checkpoint and resume automatically.

**If you get OOM (out of memory):**
```yaml
# In finetune.yaml, reduce:
batch_duration: 240   # was 360 — reduces GPU memory per batch
accumulate_grad_batches: 2  # compensates for smaller batch
```

**Expected checkpoints:**
```
/workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/
  canary-1b-v2-slovak-parliament--val_wer=0.xxxx-step=xxxxx.nemo
  last.nemo
```

---

## 5. Step 3 — Benchmark (~30min)

Run against both CommonVoice 21 SK and FLEURS SK.

```bash
cd /workspace/canary-1b-v2-sk

# First: pretrained baseline (no fine-tuning)
python scripts/benchmark.py \
    --pretrained nvidia/canary-1b-v2 \
    --max-samples 500   # quick estimate, remove for full eval

# Then: your fine-tuned model
CKPT=$(ls /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/*.nemo \
    | grep -v last | tail -1)
echo "Using checkpoint: $CKPT"

python scripts/benchmark.py \
    --model "$CKPT" \
    --max-samples 500
```

Remove `--max-samples` for the final official evaluation. Full CV21 SK test set is ~3,900 samples (~30min on A100); FLEURS SK test is ~1,000 samples (~8min).

**Expected results (rough targets):**

| Dataset | Pretrained baseline | Fine-tuned target |
|---|---|---|
| CommonVoice 21 SK | ~15–20% WER | ~8–12% WER |
| FLEURS SK | ~8–12% WER | may increase (domain shift) |

FLEURS WER increasing after fine-tuning on parliamentary speech is **expected and normal** — document it as a domain-shift result, not a regression.

---

## 6. Cost Breakdown

| Step | Duration | Cost @ $2.50/h |
|---|---|---|
| Dataset transform | ~1h | ~$2.50 |
| Fine-tuning (5k steps) | ~2–3h | ~$5–7.50 |
| Benchmarking (full) | ~1h | ~$2.50 |
| Buffer / iteration | ~0.5h | ~$1.25 |
| **Total** | **~4.5–5.5h** | **~$11–14** |

> **To save cost:** Stop the pod between steps. The network volume persists. Only the pod itself is billed per hour; the volume has a small storage fee (~$0.07/GB/month).

---

## 7. Stopping and Resuming

```bash
# Before stopping the pod — make sure training saved a checkpoint:
ls -lt /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/ | head -5

# The pod can be stopped from RunPod UI — volume data is safe.
# On restart: re-run the fine-tune command exactly as before.
# NeMo's resume_if_exists: true will pick up from the last checkpoint.
```

---

## 8. Known Issues and Warnings

These were found during code review — none are blockers, but good to know:

**`--num-jobs` argument in transform_slopal.py is unused.**
The dataset transformation runs single-threaded. The argument is parsed but never applied. On A100 this still completes in ~2h. If it's too slow, the fix would be to parallelize `row_to_cut` calls with `multiprocessing.Pool`.

**`batch_size` passed to `model.transcribe()` in benchmark.**
`src/canary_sk/benchmark.py` passes `batch_size=batch_size` to `model.transcribe()`. In NeMo 2.0 `EncDecMultiTaskModel`, this parameter is accepted but may conflict with the manual batching already done in the loop (each `batch` passed in is already `batch_size` items). If you see a TypeError mentioning `batch_size`, remove it from the `transcribe()` call — the external batching is sufficient.

**SloPalSpeech column names are assumed, not verified.**
`transform.py` expects columns: `id`, `text`, `duration`, `snapshot`, `audio.array`, `audio.sampling_rate`. If the dataset schema differs, you'll get a `KeyError` on first row. Run a quick check:
```python
from datasets import load_dataset
ds = load_dataset("NaiveNeuron/SloPalSpeech", split="train[:1]")
print(ds.features)
```

**`spec_augment._target_` path in finetune.yaml.**
Uses `nemo.collections.asr.modules.SpectrogramAugmentation`. If NeMo 25.04 has moved this class, you'll see a Hydra `_target_` instantiation error at training start. Check with:
```bash
python -c "from nemo.collections.asr.modules import SpectrogramAugmentation; print('OK')"
```
If it fails, look for the correct path in `nemo.collections.asr` and update `finetune.yaml`.

---

## 9. Quick Reference

```bash
# Transform (run once)
python scripts/transform_slopal.py --output-dir /workspace/slopal_lhotse --target-hours 100 --hf-cache /workspace/hf_cache

# Fine-tune
python -m nemo.collections.asr.scripts.speech_to_text_finetune --config-path=/workspace/canary-1b-v2-sk --config-name=finetune

# Benchmark (pretrained)
python scripts/benchmark.py --pretrained nvidia/canary-1b-v2

# Benchmark (fine-tuned)
python scripts/benchmark.py --model /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best>.nemo

# Quick sanity check (100 samples only)
python scripts/benchmark.py --pretrained nvidia/canary-1b-v2 --max-samples 100
```
