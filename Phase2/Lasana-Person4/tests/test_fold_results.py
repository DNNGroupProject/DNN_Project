"""Unit tests for fold_full_scale_results (no GPU / no model weights)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fold_full_scale_results import fold  # noqa: E402


class TestFoldFullScale(unittest.TestCase):
    def test_fold_returns_five_rows(self):
        rows = fold()
        self.assertEqual(len(rows), 5)

    def test_unet_row_present(self):
        rows = fold()
        unet = next(r for r in rows if "U-Net" in r["model"])
        self.assertNotIn(unet["dice"], ("-", "", None))
        float(unet["dice"])  # must be numeric

    def test_vanilla_segformer_from_kalana(self):
        rows = fold()
        vanilla = next(r for r in rows if "no attention" in r["model"])
        self.assertAlmostEqual(float(vanilla["dice"]), 0.8743, places=4)
        self.assertAlmostEqual(float(vanilla["aamo"]), 0.0334, places=4)
        self.assertIn("Kalana", vanilla["source"])

    def test_attention_row_is_dinura_winner(self):
        rows = fold()
        att = next(r for r in rows if "Attention Consistency Loss" in r["model"] and "Boundary" not in r["model"])
        self.assertAlmostEqual(float(att["dice"]), 0.8577, places=4)
        self.assertAlmostEqual(float(att["iou"]), 0.7508, places=4)
        self.assertAlmostEqual(float(att["aamo"]), 0.7476, places=4)
        self.assertIn("l2_1_mse", att["source"] + att["notes"])

    def test_boundary_row_is_lambda3_sweep_winner(self):
        rows = fold()
        bound = next(r for r in rows if "Boundary" in r["model"])
        self.assertAlmostEqual(float(bound["dice"]), 0.8669, places=4)
        self.assertAlmostEqual(float(bound["iou"]), 0.765, places=4)
        self.assertAlmostEqual(float(bound["aamo"]), 0.6218, places=4)
        self.assertIn("Dhinanjaya-Person5", bound["source"])

    def test_deeplab_row_present(self):
        rows = fold()
        dl = next(r for r in rows if "DeepLab" in r["model"])
        self.assertAlmostEqual(float(dl["dice"]), 0.7821, places=4)
        self.assertAlmostEqual(float(dl["iou"]), 0.6422, places=4)
        self.assertIn("train_deeplab_multiseed", dl["source"])
        self.assertNotIn("0.7369", str(dl["dice"]))


if __name__ == "__main__":
    unittest.main()
