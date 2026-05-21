# Week 1 Report

## What was completed
- Chose topic: CNN-from-scratch vs. transfer learning for natural-scene classification (C+ level).
- Selected dataset: Intel Image Classification (Kaggle, ~14k train / ~3k test, 6 classes).
- Set up repository structure (README, proposal, data instructions, src/, reports/, results/).
- Wrote `src/data_loading.py`: dataset loaders + exploratory data analysis (EDA).

## Important commits / files
- `proposal.md` — project proposal.
- `src/data_loading.py` — data loading + EDA.
- `data/README.md` — dataset download instructions.

## Experiments run
- EDA only: counted images per class, inspected a sample grid of images.

## Results so far
- Dataset loads correctly; classes are roughly balanced (to be filled with real numbers after running EDA).

## Problems / blockers
- None yet. Need a GPU for Week 2 (will use Google Colab).

## Plan for next week (Week 2)
- Implement preprocessing + train/validation split.
- Build and train the baseline CNN from scratch; report first accuracy/loss curves.
