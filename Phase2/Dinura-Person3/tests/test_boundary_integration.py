"""Unit tests for Boundary Loss wiring in train_full_scale.py (Person 5,
Week 10). No GPU, no dataset, no real transformers/SegFormer model —
transformers is stubbed so the module imports on machines that don't have
it installed (see dev_environment_notes), and grad_rollout_attention_map /
the model are replaced with fakes so only the loss-wiring logic (not the
real attention model) is under test.
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train_full_scale as T  # noqa: E402
from boundary_refinement.loss import BoundaryDiceLoss  # noqa: E402


class _FakeAttLoss:
    def __call__(self, attn_map, y0):
        return (attn_map - y0).abs().mean()


class _FakeOutputs:
    def __init__(self, logits):
        self.logits = logits


def _fake_rollout(model, x, create_graph=True):
    b, _, h, w = x.shape
    logits = torch.randn(b, 2, h // 4, w // 4, requires_grad=True)
    attn_map = torch.rand(h, w, requires_grad=True)
    return attn_map, _FakeOutputs(logits)


def _run(boundary_loss_fn=None, lambda3=0.0):
    T.grad_rollout_attention_map = _fake_rollout
    T._to_input = lambda imgs: torch.rand(len(imgs), 3, 256, 256, requires_grad=True)
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    images = np.random.rand(2, 4, 4, 3).astype("float32")
    masks = (np.random.rand(2, 256, 256) > 0.5).astype("float32")
    return T.run_epoch_attention(
        model, images, masks, opt, _FakeAttLoss(), lambda2=0.3, is_train=True,
        boundary_loss_fn=boundary_loss_fn, lambda3=lambda3,
    )


def test_run_epoch_attention_returns_five_values_with_boundary_active():
    out = _run(boundary_loss_fn=BoundaryDiceLoss(kernel_size=3), lambda3=0.2)
    assert len(out) == 5


def test_boundary_term_is_zero_when_disabled_default_args():
    """Matches run_lambda_sweep.py's SweepArgs, which carries no lambda3
    attribute at all — getattr(args, "lambda3", 0.0) must fall back to 0.0
    and produce a bit-identical (no boundary term) result."""
    out = T.run_epoch_attention(
        torch.nn.Linear(1, 1), np.random.rand(2, 4, 4, 3).astype("float32"),
        (np.random.rand(2, 256, 256) > 0.5).astype("float32"),
        torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=0.01),
        _FakeAttLoss(), lambda2=0.3, is_train=True,
    )
    assert len(out) == 5
    assert out[4] == 0.0


def test_boundary_term_nonzero_when_active():
    out = _run(boundary_loss_fn=BoundaryDiceLoss(kernel_size=3), lambda3=0.2)
    assert out[4] > 0.0


ALL_TESTS = [
    test_run_epoch_attention_returns_five_values_with_boundary_active,
    test_boundary_term_is_zero_when_disabled_default_args,
    test_boundary_term_nonzero_when_active,
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
