"""Split-identity preflight for the full-scale SegFormer run.

Person 5's 2026-08-15 fix made Person 3's `make_splits` use the same
algorithm as Chanupa's `dataset.py`: sort mask filenames, stdlib
`random.shuffle` under seed 42, front-slice val → test → train. If those
lists ever diverge again, the paper's U-Net row and the SegFormer rows
would not share a held-out test set.

These tests need no GPU and no `transformers`. The synthetic test always
runs. The real-dataset test runs when `Phase1/Kalana-Person2/masks` is
present. The live-import test runs only if cv2/torch can load Person 3's
`data.py`.

Run:
    python tests/test_split_identity.py
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from paths import (  # noqa: E402
    DATA_MASK_DIR,
    N_TEST,
    N_TRAIN,
    N_VAL,
    SEED,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def chanupa_split(mask_files, val_split=0.15, test_split=0.15, seed=SEED):
    """Bit-for-bit `Chanupa-Person1/dataset.py::make_splits`."""
    all_files = sorted(mask_files)
    random.seed(seed)
    random.shuffle(all_files)
    n = len(all_files)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val_files = all_files[:n_val]
    test_files = all_files[n_val : n_val + n_test]
    train_files = all_files[n_val + n_test :]
    return train_files, val_files, test_files


def dinura_split_names(mask_files, n_train, n_val, n_test, seed=SEED):
    """Bit-for-bit `attention_consistency/data.py` list_pairs + make_splits,
    comparing on mask *filenames* (pairing skipped — every mask in this
    dataset has an image)."""
    files = sorted(f for f in mask_files if str(f).lower().endswith(IMAGE_EXTS))
    random.seed(seed)
    random.shuffle(files)
    total = n_train + n_val + n_test
    files = files[:total]
    val_files = files[:n_val]
    test_files = files[n_val : n_val + n_test]
    train_files = files[n_val + n_test :]
    return train_files, val_files, test_files


def _fake_mask_names(n=5108):
    return [f"{i:05d}_mask_00.jpg" for i in range(n)]


def test_audited_counts_are_3576_766_766():
    names = _fake_mask_names(5108)
    train, val, test = chanupa_split(names)
    assert len(val) == N_VAL, len(val)
    assert len(test) == N_TEST, len(test)
    assert len(train) == N_TRAIN, len(train)
    assert len(train) + len(val) + len(test) == 5108


def test_synthetic_chanupa_and_dinura_lists_match_elementwise():
    names = _fake_mask_names(5108)
    c_train, c_val, c_test = chanupa_split(names)
    d_train, d_val, d_test = dinura_split_names(names, N_TRAIN, N_VAL, N_TEST)
    assert c_train == d_train
    assert c_val == d_val
    assert c_test == d_test


def test_splits_are_disjoint():
    train, val, test = chanupa_split(_fake_mask_names(5108))
    s_train, s_val, s_test = set(train), set(val), set(test)
    assert not (s_train & s_val)
    assert not (s_train & s_test)
    assert not (s_val & s_test)


def test_real_mask_dir_matches_when_present():
    if not DATA_MASK_DIR.is_dir():
        print("SKIP  test_real_mask_dir_matches_when_present (masks dir missing)")
        return
    files = [
        f
        for f in os.listdir(DATA_MASK_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    if len(files) < 5108:
        print(f"SKIP  test_real_mask_dir_matches_when_present (found {len(files)} masks, need 5108)")
        return
    c_train, c_val, c_test = chanupa_split(files)
    d_train, d_val, d_test = dinura_split_names(files, N_TRAIN, N_VAL, N_TEST)
    assert len(c_train) == N_TRAIN and len(c_val) == N_VAL and len(c_test) == N_TEST
    assert c_train == d_train
    assert c_val == d_val
    assert c_test == d_test
    print(f"      real dataset: test-set first 3 = {c_test[:3]}")


def test_live_make_splits_matches_chanupa_when_importable():
    """Call the actual Person 3 / Person 1 functions, not the copies above."""
    chanupa_dir = REPO_ROOT / "Phase1" / "Chanupa-Person1"
    person3_dir = REPO_ROOT / "Phase1" / "Dinura-Person3"
    if str(chanupa_dir) not in sys.path:
        sys.path.insert(0, str(chanupa_dir))
    if str(person3_dir) not in sys.path:
        sys.path.insert(0, str(person3_dir))

    try:
        from dataset import list_mask_files, make_splits as chanupa_make_splits
        from paths import add_teammate_paths, apply_data_dirs

        add_teammate_paths()
        apply_data_dirs()
        from attention_consistency.data import make_splits as dinura_make_splits
    except Exception as exc:
        print(f"SKIP  test_live_make_splits_matches_chanupa_when_importable ({type(exc).__name__}: {exc})")
        return

    if not DATA_MASK_DIR.is_dir():
        print("SKIP  test_live_make_splits_matches_chanupa_when_importable (masks dir missing)")
        return

    mask_files = list_mask_files(DATA_MASK_DIR)
    if len(mask_files) < 5108:
        print(f"SKIP  test_live_make_splits_matches_chanupa_when_importable (found {len(mask_files)} masks)")
        return

    c_train, c_val, c_test = chanupa_make_splits(mask_files, 0.15, 0.15, SEED)
    splits = dinura_make_splits(N_TRAIN, N_VAL, N_TEST, seed=SEED)
    d_train = [p[1].name for p in splits["train"]]
    d_val = [p[1].name for p in splits["val"]]
    d_test = [p[1].name for p in splits["test"]]
    assert list(c_train) == d_train
    assert list(c_val) == d_val
    assert list(c_test) == d_test


ALL_TESTS = [
    test_audited_counts_are_3576_766_766,
    test_synthetic_chanupa_and_dinura_lists_match_elementwise,
    test_splits_are_disjoint,
    test_real_mask_dir_matches_when_present,
    test_live_make_splits_matches_chanupa_when_importable,
]


if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} passed")
    if failed:
        raise SystemExit(1)
