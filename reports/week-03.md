# Week 3 Report

## What was completed
- Added data augmentation (random flip, rotation, color jitter) applied to the
  training set only (`src/data_loading.py`, augment=True).
- Trained the same baseline CNN WITH augmentation (`src/train_augmented.py`) to
  compare against the Week 2 baseline (no augmentation).
- Implemented transfer learning with a pretrained ResNet-18 (`src/model.py`,
  `build_resnet18`) and trained it (`src/train_resnet.py`, 224x224 + ImageNet
  normalization, lr=1e-4).
- Built a comparison script (`src/compare.py`) that collects all results into a table.

## Important commits / files
- `src/train_augmented.py`, `src/train_resnet.py`, `src/compare.py`
- `results/augmented_*`, `results/resnet_*`, `results/comparison_table.md`

## Experiments run
- Baseline CNN (no aug) — from Week 2.
- Baseline CNN + augmentation — 10 epochs, Adam lr=1e-3.
- ResNet-18 transfer learning — 5 epochs, Adam lr=1e-4.

## Results so far (fill from results/comparison_table.md)
| Model | Test accuracy | Macro F1 |
| ----- | ------------- | -------- |
| Baseline CNN (no augmentation)   | 84.6 % | 0.849 |
| Baseline CNN + augmentation      | 81.8 % | 0.820 |
| ResNet-18 (transfer learning)    | 92.7 % | 0.928 |

- Augmentation effect: train/val gap (overfitting) ____ (increased / decreased).
- Best model: ResNet-18 . Why it wins: it reuses features learned on ImageNet (1.2M images),
  so it needs fewer epochs and generalizes better than a small CNN trained from scratch.

## Problems / blockers
- ResNet training is slower per epoch (bigger images 224x224) — kept epochs low (5).

## Plan for next week (Week 4)
- Error analysis on the best model (which classes still get confused and why).
- Write the final report (sections: dataset, preprocessing, models, training, results, error analysis, limitations, conclusion).
- Prepare the short presentation/demo.
