import hashlib
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
        bucket = int(hashlib.md5(c.id.encode()).hexdigest(), 16) % 1000
        by_session[bucket].append(c)

    sessions = list(by_session.keys())
    rng.shuffle(sessions)

    train, dev, test = [], [], []
    dev_budget = dev_hours * 3600
    test_budget = test_hours * 3600

    dev_running = 0.0
    test_running = 0.0
    for s in sessions:
        session_dur = sum(c.duration for c in by_session[s])
        if dev_running < dev_budget:
            dev.extend(by_session[s])
            dev_running += session_dur
        elif test_running < test_budget:
            test.extend(by_session[s])
            test_running += session_dur
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
