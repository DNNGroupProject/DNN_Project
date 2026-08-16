# Full-scale SegFormer (Phase 2 / Kalana-Person2)

Person 4 formulas (`metrics.py` / `aamo.py` / `efficiency.py`). Smoke-scale rows were not copied here.

| Model | Dice | IoU | F1 | AAMO | Params | GFLOPs | FPS |
|-------|------|-----|----|------|--------|--------|-----|
| SegFormer-B0 (no attention loss) | 0.8743 | 0.7766 | 0.8743 | 0.0334 | 3714658 | 1.692 | 84.52 |
| SegFormer-B0 + Attention Consistency Loss | 0.869 | 0.7684 | 0.869 | 0.5752 | 3714658 | 1.692 | 100.81 |
