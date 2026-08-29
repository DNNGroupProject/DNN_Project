"""Unit tests for Person 3 Phase 2 sweep helpers — no GPU, no dataset."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from aggregate_sweep import pick_winner  # noqa: E402
from paths import run_tag  # noqa: E402


def test_run_tag_strips_trailing_zeros():
    assert run_tag(0.1, "mse") == "l2_0.1_mse"
    assert run_tag(0.3, "mse") == "l2_0.3_mse"
    assert run_tag(1.0, "kl") == "l2_1_kl"
    assert run_tag(0.5, "mse") == "l2_0.5_mse"


def test_pick_winner_prefers_higher_aamo():
    rows = [
        {"run_tag": "l2_0.1_mse", "lambda2": 0.1, "att_mode": "mse",
         "test_aamo": 0.40, "test_dice": 0.87, "best_val_dice": 0.79},
        {"run_tag": "l2_0.3_mse", "lambda2": 0.3, "att_mode": "mse",
         "test_aamo": 0.5752, "test_dice": 0.869, "best_val_dice": 0.7902},
        {"run_tag": "l2_0.5_mse", "lambda2": 0.5, "att_mode": "mse",
         "test_aamo": 0.50, "test_dice": 0.88, "best_val_dice": 0.80},
    ]
    assert pick_winner(rows)["run_tag"] == "l2_0.3_mse"


def test_pick_winner_breaks_aamo_tie_with_dice():
    rows = [
        {"run_tag": "a", "lambda2": 0.1, "att_mode": "mse",
         "test_aamo": 0.5, "test_dice": 0.86, "best_val_dice": 0.7},
        {"run_tag": "b", "lambda2": 0.3, "att_mode": "mse",
         "test_aamo": 0.5, "test_dice": 0.88, "best_val_dice": 0.7},
    ]
    assert pick_winner(rows)["run_tag"] == "b"


def test_pick_winner_prefers_mse_on_exact_tie():
    rows = [
        {"run_tag": "kl", "lambda2": 0.3, "att_mode": "kl",
         "test_aamo": 0.5, "test_dice": 0.87, "best_val_dice": 0.7},
        {"run_tag": "mse", "lambda2": 0.3, "att_mode": "mse",
         "test_aamo": 0.5, "test_dice": 0.87, "best_val_dice": 0.7},
    ]
    assert pick_winner(rows)["run_tag"] == "mse"


def test_pick_winner_falls_back_to_val_dice():
    rows = [
        {"run_tag": "x", "lambda2": 0.1, "att_mode": "mse",
         "test_aamo": "", "test_dice": "", "best_val_dice": 0.70},
        {"run_tag": "y", "lambda2": 0.5, "att_mode": "mse",
         "test_aamo": "", "test_dice": "", "best_val_dice": 0.80},
    ]
    assert pick_winner(rows)["run_tag"] == "y"


def test_pick_winner_empty():
    assert pick_winner([]) is None


ALL_TESTS = [
    test_run_tag_strips_trailing_zeros,
    test_pick_winner_prefers_higher_aamo,
    test_pick_winner_breaks_aamo_tie_with_dice,
    test_pick_winner_prefers_mse_on_exact_tie,
    test_pick_winner_falls_back_to_val_dice,
    test_pick_winner_empty,
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
