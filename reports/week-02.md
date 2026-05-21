# Week 2 Report

## What was completed
- Implemented the baseline CNN from scratch (`src/model.py`, class `SimpleCNN`):
  3 conv blocks (Conv-ReLU-MaxPool) + adaptive pooling + 2 fully-connected layers with dropout.
- Built a reusable training/evaluation engine (`src/engine.py`): train loop, validation,
  full prediction, and loss/accuracy plotting.
- Wrote `src/train_baseline.py` to train end-to-end, evaluate on the untouched test set,
  and save curves, confusion matrix, and a metrics report.
- Used the train/val/test split from Week 1 (80/20 split of seg_train; seg_test kept for final test only).

## Important commits / files
- `src/model.py` — baseline CNN architecture.
- `src/engine.py` — training and evaluation functions.
- `src/train_baseline.py` — baseline training script.
- `results/baseline_curves.png`, `results/baseline_confusion.png`, `results/baseline_metrics.txt`.

## Experiments run
- Trained SimpleCNN for 10 epochs, Adam optimizer (lr=1e-3), Cross-Entropy loss, batch size 32.

## Results so far
- Baseline test accuracy: ____ %   (fill in from results/baseline_metrics.txt)
- Validation accuracy plateaued around epoch ____.
- Most confused classes (from confusion matrix): ____ vs ____.

## Problems / blockers
- (e.g. signs of overfitting: train accuracy >> val accuracy) — to confirm from curves.

## Plan for next week (Week 3)
- Add transfer learning with a pretrained ResNet-18 and compare against this baseline.
- Add data augmentation and measure its effect on the baseline.
- Build a comparison table of all setups.
