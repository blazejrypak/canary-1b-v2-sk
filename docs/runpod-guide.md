# RunPod Guide — Canary-1B-v2 Slovak Fine-tuning

This is a step-by-step guide for first-timers. It covers everything from creating a RunPod account to downloading your fine-tuned model.

**What you'll do:**
1. Create a pod, set up the environment
2. Run the dataset transform script (~1–2h, CPU work)
3. Fine-tune Canary-1B-v2 (~3–4h, GPU)
4. Run benchmark evaluation (~30min)

**Total cost: ~$12–18.** Total time with downloads: ~6–8h.

---

## 1. What Pod Do You Need?

There are two steps with very different requirements. You have two options:

### Disk usage breakdown

Before picking a volume size, here is what actually occupies space on the persistent volume (`/workspace`):

| Item | Size | Notes |
|---|---|---|
| HF parquet cache (`hf_cache/`) | ~60 GB | SloPalSpeech full dataset; cached by `--hf-cache` for resumability |
| Output shards (`slopal_lhotse/`) | ~6 GB | 100h of audio re-encoded as FLAC |
| Training checkpoints (`exp/`) | ~8 GB | 4 checkpoints × ~2 GB each |
| Code repo | ~0.1 GB | |
| **Total** | **~74 GB** | |

The NeMo container image (~25 GB) and the downloaded `nvidia/canary-1b-v2` model (~4 GB) go on the **container disk** (separate 50 GB), not the volume.

**100 GB volume is enough** — gives ~26 GB of headroom. 200 GB was over-conservative.

> **Want to use even less?** Skip `--hf-cache /workspace/hf_cache` from the transform command. The parquet files won't be cached on the volume (only ~14 GB used total), but if the script crashes mid-run you'll need to re-download from HF (~60 GB, ~30–60 min).

---

### Option A — One pod for everything (recommended for first run)

Use one A100 pod for all three steps. Simpler setup, slightly higher cost.

| Setting | Value |
|---|---|
| GPU | **A100 SXM 80GB** |
| Container image | `nvcr.io/nvidia/nemo:25.04` |
| Container disk | 50 GB |
| Volume disk | **100 GB** (persistent — survives pod stops) |

Cost: ~$2.50/h × 5–7h active = **~$12–18 total**.

### Option B — Two pods (save ~$2–3)

Run the dataset transform on a cheap CPU pod first, then use the A100 only for fine-tuning.

**Pod 1 — dataset transform (CPU):**

| Setting | Value |
|---|---|
| GPU | None — use **CPU pod** |
| CPU | 8+ vCPU, 32 GB RAM |
| Container image | `python:3.11-slim` or any Python image |
| Volume disk | **100 GB** |

Cost: ~$0.10/h × 1–2h = **~$0.10–0.20**

**Pod 2 — fine-tuning (GPU):**

| Setting | Value |
|---|---|
| GPU | **A100 SXM 80GB** (preferred) or A40 48GB |
| Container image | `nvcr.io/nvidia/nemo:25.04` |
| Container disk | 50 GB |
| Volume disk | **same 100 GB volume** (attach the one from Pod 1) |

> If you use an A40 48GB instead of A100 80GB, reduce `batch_duration: 180` in `finetune.yaml` and set `accumulate_grad_batches: 2` to compensate.

**This guide uses Option A** (one A100 pod). If you choose Option B, the steps are the same — just skip the GPU-only steps on the CPU pod.

---

## 2. Prerequisites

Before touching RunPod, make sure you have:

- [ ] **HuggingFace account** with access to:
  - `NaiveNeuron/SloPalSpeech` (check if gated — request access if needed)
  - `mozilla-foundation/common_voice_21_0` (requires HF login)
  - `google/fleurs` (public)
- [ ] **HF token** with read access: https://huggingface.co/settings/tokens
  - Create one → "New token" → "Read" → copy it somewhere safe
- [ ] **This repo** accessible (GitHub private repo with deploy key, or just clone via HTTPS with your GitHub token)
- [ ] **RunPod account** with billing enabled: https://www.runpod.io

---

## 3. Create Your RunPod Account and Add Credits

1. Go to https://www.runpod.io and sign up
2. Go to **Billing** → add a credit card → deposit **$25** (enough for this project with buffer)
3. That's it — you pay only for what you use

---

## 4. Create the Network Volume First

> **Important:** Network volumes are region-locked. The volume must be in the same datacenter as your GPU pod or RunPod won't let you attach it. So create the volume first, in the right region.

### Step 1 — Find a datacenter with A100 SXM 80GB available

1. Go to **Pods** → **+ Deploy** → **GPU Cloud**
2. Search for **A100 SXM 80GB**, filter by **Secure Cloud**
3. Note which datacenters show availability (e.g. `EU-RO-1`, `US-TX-3`, `CA-MTL-3`) — availability changes daily
4. Pick one. **Don't deploy yet** — close or go back.

### Step 2 — Create the network volume in that datacenter

1. Go to **Storage** (left sidebar) → **+ New Network Volume**
2. Set:

| Field | Value |
|---|---|
| Datacenter | **same one you picked above** |
| Size | 100 GB |
| Name | `canary-sk-volume` (or anything) |

3. Click **Create** — takes a few seconds.

### Step 3 — Create the pod and attach the volume

1. Go back to **Pods** → **+ Deploy** → **GPU Cloud**
2. Search for **A100 SXM 80GB**, Secure Cloud, **same datacenter**
3. Click **Customize Deployment** and set:

| Field | Value |
|---|---|
| GPU Count | 1 |
| Container image | `nvcr.io/nvidia/nemo:25.04` |
| Container disk | 50 GB |
| Network Volume | select `canary-sk-volume` from the dropdown |
| Volume mount path | `/workspace` |

Under **Environment Variables**, add:
```
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
```
(paste your HuggingFace token)

4. Click **Deploy** — the pod will start in 1–5 minutes

> If the datacenter you picked runs out of A100s between step 1 and step 3, pick the next available datacenter and create a new volume there. The empty volume from step 2 can be deleted — you haven't stored anything in it yet.

---

## 5. Connect via SSH

### 5a. Add your SSH public key to RunPod

1. On your Mac, open Terminal and check if you have an SSH key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   If you get "No such file or directory", create one:
   ```bash
   ssh-keygen -t ed25519 -C "your@email.com"
   # press Enter 3 times to accept defaults and skip passphrase
   cat ~/.ssh/id_ed25519.pub
   ```
2. Copy the output (starts with `ssh-ed25519 AAAA...`)
3. In RunPod: go to **Settings** → **SSH Public Keys** → paste it → Save

### 5b. Connect

Once the pod is running:
1. Click your pod → click **Connect** → copy the SSH command
2. It looks like: `ssh root@<ip-address> -p <port>`
3. Paste it in your Mac Terminal and press Enter
4. Type `yes` when asked about the host fingerprint

You are now inside the pod.

---

## 6. tmux — Keep Jobs Running When You Disconnect

> **Important:** Without tmux, your job dies the moment your SSH connection drops (wifi blip, laptop sleeps, etc.). Always use tmux for long-running jobs.

### Start a tmux session (do this first, every time you connect)

```bash
tmux new -s main
```

You'll see a green bar at the bottom — you're inside tmux now.

### Detach (leave job running, go back to your Mac)

Press `Ctrl+B`, then `D` (hold Ctrl, press B, release both, press D).

The job keeps running on the pod. You can close your terminal.

### Reattach (come back to your running job)

SSH into the pod again, then:
```bash
tmux attach -t main
```

### Other useful tmux commands

```bash
tmux ls                  # list sessions
tmux new -s logs         # create a second session for watching logs
Ctrl+B, [                # scroll mode — use arrow keys / PgUp to scroll output
q                        # exit scroll mode
Ctrl+B, D                # detach from any session
```

---

## 7. One-time Pod Setup

Do this once after you first connect (inside the tmux session):

```bash
# 1. Go to the persistent volume
cd /workspace

# 2. Clone the repo
#    Option A — HTTPS (simpler, needs your GitHub token)
git clone https://github.com/YOUR_USERNAME/canary-1b-v2-sk.git
#    Option B — SSH (needs deploy key setup, see below)
# git clone git@github.com:YOUR_USERNAME/canary-1b-v2-sk.git

cd canary-1b-v2-sk

# 3. Install dependencies
#    (nemo_toolkit is already in the container — this adds lhotse, datasets, jiwer)
pip install -r requirements.txt

# 4. Install this repo's package
pip install -e .

# 5. Verify everything is installed
python -c "import lhotse; import datasets; import nemo; print('All OK')"

# 6. Verify HF token is working
python -c "from huggingface_hub import whoami; print(whoami()['name'])"

# 7. Verify dataset schema (quick check — downloads ~10 MB, not the full dataset)
python -c "
from datasets import load_dataset
ds = load_dataset('NaiveNeuron/SloPalSpeech', split='train[:2]', streaming=False)
print('Columns:', list(ds.features.keys()))
print('First row keys:', list(ds[0].keys()))
"
# Expected columns: id, text, duration, snapshot, audio
```

> **If git clone via HTTPS asks for password:** use your GitHub username + a Personal Access Token (not your GitHub password). Create one at GitHub → Settings → Developer settings → Personal access tokens → Classic → select `repo` scope.

---

## 8. Step 1 — Transform SloPalSpeech (~1–2h)

This downloads the dataset and writes Lhotse Shar shards. It is CPU-only work — the GPU is idle during this step.

```bash
cd /workspace/canary-1b-v2-sk

python scripts/transform_slopal.py \
    --output-dir /workspace/slopal_lhotse \
    --target-hours 100 \
    --hf-cache /workspace/hf_cache
```

**What to expect:**

```
>>> Pass 1: streaming metadata (audio column skipped)...
    50,000 rows scanned (125 h accumulated)...
    ...
    ~350,000 segments found (~2806 h total).
>>> Stratified sampling for ~100.0 h...
    Selected ~12,500 / ~350,000 segments (3.5%).
>>> Computing train/dev/test assignment...
    train=~11,600 (94.0 h)  dev=~450 (3.0 h)  test=~450 (3.0 h)
>>> Pass 2: streaming audio and writing shards...
    1,000 cuts written | ...
    ...
>>> ALL DONE
    Output: /workspace/slopal_lhotse
```

Pass 1 takes ~30–60 min (streaming metadata from HF). Pass 2 takes ~30–60 min (decoding and encoding ~12k audio segments as FLAC).

**Verify the output:**
```bash
ls /workspace/slopal_lhotse/train/
# Should show: cuts.000000.jsonl.gz, recording.000000.tar, ...

ls /workspace/slopal_lhotse/dev/
ls /workspace/slopal_lhotse/test/
```

> **If the download stalls or the pod is interrupted:** re-run the same command. The `datasets` library caches parquet files in `--hf-cache`. Pass 1 will resume from cache; Pass 2 will restart from row 0 but the shard files will be overwritten cleanly.

---

## 9. Step 2 — Fine-tune (~3–4h)

Now the GPU is used. The `finetune.yaml` is already configured for 100h of Slovak parliamentary speech.

```bash
cd /workspace/canary-1b-v2-sk

python -m nemo.collections.asr.scripts.speech_to_text_finetune \
    --config-path=/workspace/canary-1b-v2-sk \
    --config-name=finetune
```

**What happens on first run:**
1. NeMo downloads `nvidia/canary-1b-v2` from HF Hub to NeMo cache (~4 GB, one-time, takes ~5 min)
2. Loads pretrained weights
3. Trains for 5,000 steps — validates every 1,000 steps, saves top-3 checkpoints by `val_wer`

**Watch training in a second tmux window:**

Open a new tmux pane: press `Ctrl+B`, then `"` (splits horizontally) or `%` (splits vertically).

```bash
# Watch the NeMo log:
tail -f /workspace/exp/canary-1b-v2-slovak-parliament/nemo_log_globalrank-0_localrank-0.txt

# Or watch GPU usage:
watch -n 2 nvidia-smi
```

Switch between panes: `Ctrl+B`, then arrow key.

**Expected checkpoints when done:**
```
/workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/
  canary-1b-v2-slovak-parliament--val_wer=0.xxxx-step=xxxxx.nemo
  last.nemo
```

**If the pod is interrupted mid-training:** re-run the exact same command. `resume_if_exists: true` in `finetune.yaml` makes NeMo automatically find the last checkpoint and resume.

**If you get OOM (out of memory) error:**
```yaml
# Edit finetune.yaml on the pod:
nano /workspace/canary-1b-v2-sk/finetune.yaml

# Reduce these two values:
batch_duration: 180        # was 360
accumulate_grad_batches: 2 # compensates for smaller batch
```

---

## 10. Step 3 — Benchmark (~30–60 min)

Compare your fine-tuned model against the pretrained baseline.

```bash
cd /workspace/canary-1b-v2-sk

# Baseline (pretrained, no fine-tuning):
python scripts/benchmark.py \
    --pretrained nvidia/canary-1b-v2 \
    --max-samples 500

# Find your best checkpoint:
ls /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/*.nemo | grep -v last

# Fine-tuned model:
python scripts/benchmark.py \
    --model /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best>.nemo \
    --max-samples 500
```

Remove `--max-samples 500` for the full official evaluation (takes ~40 min).

**Expected results:**

| Dataset | Pretrained baseline | Fine-tuned target |
|---|---|---|
| CommonVoice 21 SK | ~15–20% WER | ~8–12% WER |
| FLEURS SK | ~8–12% WER | may increase (domain shift — expected) |

---

## 11. Download Your Model to Your Mac

After benchmarking, copy the best checkpoint off the pod before stopping it.

**Option A — scp (simplest):**
```bash
# On your Mac (not on the pod):
scp -P <port> root@<ip>:/workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best>.nemo ./
```

**Option B — upload to HuggingFace:**
```bash
# On the pod:
pip install huggingface_hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='/workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best>.nemo',
    path_in_repo='canary-1b-v2-sk.nemo',
    repo_id='YOUR_HF_USERNAME/canary-1b-v2-sk',
    repo_type='model',
    token='hf_xxxx',
)
"
```

---

## 12. Cost and Time Breakdown

| Step | Duration | Cost @ $2.50/h A100 |
|---|---|---|
| Pod setup + downloads | ~30 min | ~$1.25 |
| Dataset transform | ~1–2h | ~$2.50–5.00 |
| Fine-tuning (5k steps) | ~3–4h | ~$7.50–10.00 |
| Benchmarking | ~30–60 min | ~$1.25–2.50 |
| Buffer | ~30 min | ~$1.25 |
| **Total** | **~5.5–8h** | **~$13–20** |

> **To save money:** Stop the pod between steps — only the pod is billed per hour. The 100 GB volume has a storage fee of ~$0.07/GB/month = ~$7/month, so don't leave it sitting for weeks.

---

## 13. Stopping and Resuming the Pod

**Before stopping — make sure your data is on the volume (not container disk):**
```bash
ls /workspace/slopal_lhotse/    # dataset shards
ls /workspace/hf_cache/         # HF download cache (saves re-downloading)
ls /workspace/exp/              # training checkpoints
ls /workspace/canary-1b-v2-sk/  # your code
```

Everything under `/workspace/` is on the persistent volume and survives pod stops.

**Stop the pod:**
- RunPod UI → your pod → **Stop** (not Terminate!)
- "Terminate" deletes the pod AND the container disk. The volume survives.
- "Stop" keeps the pod configuration. You can restart it.

**Resume:**
- RunPod UI → your pod → **Start**
- SSH in again, `tmux attach -t main` (or create a new session with `tmux new -s main`)
- Re-run whatever step was interrupted — all three steps are safe to re-run

---

## 14. Known Issues and Workarounds

**`batch_size` TypeError in benchmark.py**

`benchmark.py` passes `batch_size=batch_size` to `model.transcribe()`. In NeMo 2.0 this may conflict with the manual batching in the loop. If you see:
```
TypeError: transcribe() got an unexpected keyword argument 'batch_size'
```
Open `src/canary_sk/benchmark.py` and remove the `batch_size=batch_size` argument from the `model.transcribe()` call.

**SloPalSpeech column names changed**

The transform script expects columns: `id`, `text`, `duration`, `snapshot`, `audio`. If the dataset schema changed and you get a `KeyError`, run:
```bash
python -c "
from datasets import load_dataset
ds = load_dataset('NaiveNeuron/SloPalSpeech', split='train[:1]')
print(ds.features)
"
```
then adjust column names in `src/canary_sk/transform.py`.

**SpectrogramAugmentation import error**

If fine-tuning fails with a Hydra `_target_` error mentioning `SpectrogramAugmentation`:
```bash
python -c "from nemo.collections.asr.modules import SpectrogramAugmentation; print('OK')"
```
If it fails, find the correct path:
```bash
python -c "import nemo.collections.asr.modules as m; print([x for x in dir(m) if 'Augment' in x])"
```
Then update `spec_augment._target_` in `finetune.yaml`.

**HF dataset download stalls**

The HF parquet files are large. If downloading stalls for >10 min with no progress, kill the script (`Ctrl+C`) and rerun it — the cache picks up where it left off.

---

## 15. Quick Reference

```bash
# Connect to pod
ssh root@<ip> -p <port>

# Start/reattach tmux
tmux new -s main          # first time
tmux attach -t main       # coming back

# Setup (once)
cd /workspace/canary-1b-v2-sk && pip install -r requirements.txt && pip install -e .

# Transform dataset
python scripts/transform_slopal.py --output-dir /workspace/slopal_lhotse --target-hours 100 --hf-cache /workspace/hf_cache

# Fine-tune
python -m nemo.collections.asr.scripts.speech_to_text_finetune --config-path=/workspace/canary-1b-v2-sk --config-name=finetune

# Benchmark — pretrained baseline
python scripts/benchmark.py --pretrained nvidia/canary-1b-v2

# Benchmark — fine-tuned (replace with your checkpoint path)
python scripts/benchmark.py --model /workspace/exp/canary-1b-v2-slovak-parliament/checkpoints/<best>.nemo

# Watch GPU
watch -n 2 nvidia-smi

# Watch training log
tail -f /workspace/exp/canary-1b-v2-slovak-parliament/nemo_log_globalrank-0_localrank-0.txt
```
