"""Unit tests for Person 5's Boundary Loss λ3-sweep helpers — pure string/
selection logic only, no GPU or dataset. `run_boundary_sweep` transitively
imports Dinura's train_full_scale.py, which needs `transformers` for the
real SegFormer model even though nothing here trains one — stub it so this
test runs on machines without it installed (see dev_environment_notes).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.SegformerConfig = object
    _fake_tf.SegformerForSemanticSegmentation = object
    sys.modules["transformers"] = _fake_tf

from aggregate_boundary_sweep import _parse_tag, pick_winner  # noqa: E402
from run_boundary_sweep import boundary_tag  # noqa: E402


def test_boundary_tag_matches_train_full_scale_naming():
    assert boundary_tag(1.0, "mse", 0.2) == "l2_1_mse_bnd0.2"
    assert boundary_tag(1.0, "mse", 0.1) == "l2_1_mse_bnd0.1"
    assert boundary_tag(0.3, "kl", 0.5) == "l2_0.3_kl_bnd0.5"


def test_parse_tag_round_trips_boundary_tag():
    tag = boundary_tag(1.0, "mse", 0.2)
    meta = _parse_tag(tag)
    assert meta == {"run_tag": tag, "lambda2": 1.0, "att_mode": "mse", "lambda3": 0.2}


def test_parse_tag_rejects_plain_lambda2_tag():
    # A λ2-sweep tag (no _bnd suffix) must not be picked up by the boundary aggregator.
    assert _parse_tag("l2_1_mse") is None


def test_pick_winner_prefers_higher_dice_not_aamo():
    """Differs from Dinura's aggregate_sweep.pick_winner (AAMO-first) —
    L_boundary targets Dice/IoU, not attention faithfulness."""
    rows = [
        {"run_tag": "l2_1_mse_bnd0.1", "lambda3": 0.1, "test_dice": 0.86, "test_iou": 0.75, "test_aamo": 0.80},
        {"run_tag": "l2_1_mse_bnd0.2", "lambda3": 0.2, "test_dice": 0.88, "test_iou": 0.76, "test_aamo": 0.70},
        {"run_tag": "l2_1_mse_bnd0.5", "lambda3": 0.5, "test_dice": 0.87, "test_iou": 0.77, "test_aamo": 0.75},
    ]
    assert pick_winner(rows)["run_tag"] == "l2_1_mse_bnd0.2"


def test_pick_winner_breaks_dice_tie_with_iou():
    rows = [
        {"run_tag": "a", "lambda3": 0.1, "test_dice": 0.87, "test_iou": 0.75, "test_aamo": 0.7},
        {"run_tag": "b", "lambda3": 0.2, "test_dice": 0.87, "test_iou": 0.78, "test_aamo": 0.7},
    ]
    assert pick_winner(rows)["run_tag"] == "b"


def test_pick_winner_falls_back_to_val_dice_when_no_test_eval():
    rows = [
        {"run_tag": "x", "lambda3": 0.1, "test_dice": "", "test_iou": "", "best_val_dice": 0.70},
        {"run_tag": "y", "lambda3": 0.5, "test_dice": "", "test_iou": "", "best_val_dice": 0.80},
    ]
    assert pick_winner(rows)["run_tag"] == "y"


def test_pick_winner_empty():
    assert pick_winner([]) is None


ALL_TESTS = [
    test_boundary_tag_matches_train_full_scale_naming,
    test_parse_tag_round_trips_boundary_tag,
    test_parse_tag_rejects_plain_lambda2_tag,
    test_pick_winner_prefers_higher_dice_not_aamo,
    test_pick_winner_breaks_dice_tie_with_iou,
    test_pick_winner_falls_back_to_val_dice_when_no_test_eval,
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
