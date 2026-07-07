project1/
├── config.py      – all hyperparameters & paths (single place to change things)
├── dataset.py     – ForestDataset + train/val/test split (80/10/10)
├── model.py       – U-Net baseline (31M parameters)
├── losses.py      – BCE + Dice combined loss
├── metrics.py     – IoU, Dice, Pixel Accuracy, Precision, Recall, F1
├── train.py       – full training loop → checkpoints + CSV log + curves PNG
└── evaluate.py    – test-set evaluation → metrics report + prediction grid PNG


Architecture – U-Net (standard research baseline)
The model follows Ronneberger et al. (2015) with BatchNorm added:

Input (3×256×256)
  → Encoder: 4 × [DoubleConv + MaxPool]  (channels: 3→64→128→256→512)
  → Bottleneck: DoubleConv (512→1024)
  → Decoder: 4 × [ConvTranspose2d + skip-cat + DoubleConv]
  → Head: 1×1 Conv
Output (1×256×256) logits  →  sigmoid → binary mask

