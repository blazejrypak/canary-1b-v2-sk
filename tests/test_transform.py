from types import SimpleNamespace

from canary_sk.transform import assign_splits, is_valid, split_cuts, stratified_indices


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


# ---- assign_splits ----


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
    kept = _make_kept(1000, duration=10.0)  # 10000s total — enough for dev+test at 1h each
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
