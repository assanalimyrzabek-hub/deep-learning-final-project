# Natural Scene Image Classification — CNN vs. Transfer Learning

Deep Learning final project (Narxoz University, Spring 2026). Target level: **C+**.

The goal is to classify natural-scene photos into 6 classes
(*buildings, forest, glacier, mountain, sea, street*) and to compare a CNN trained
from scratch against a pretrained **ResNet-18**, plus measure the effect of data augmentation.

## Results (test set, ~3000 images)

| Model | Test accuracy | Macro F1 |
| ----- | ------------- | -------- |
| ResNet-18 (transfer learning)    | **92.6%** | 0.928 |
| Baseline CNN (no augmentation)   | 84.6% | 0.849 |
| Baseline CNN + augmentation      | 81.8% | 0.820 |

See `final-report.md` for the full write-up and `presentation.pptx` for the slides.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── proposal.md            # project proposal (Week 1)
├── final-report.md        # final report (Week 4)
├── presentation.pptx      # defense slides
├── presentation_outline.md
├── data/
│   └── README.md          # how to download the dataset (data itself is NOT committed)
├── src/                   # data loading, models, training, evaluation, error analysis
├── reports/               # weekly progress reports (week-01 ... week-04)
└── results/               # saved figures, metrics, comparison table
```

## Dataset

Intel Image Classification — https://www.kaggle.com/datasets/puneet6060/intel-image-classification

See `data/README.md` for download instructions. The raw dataset is **not** stored in
this repository (per course rules — no large raw data in git).

## Setup

```
pip install -r requirements.txt
```

Then download the dataset into `data/` as described in `data/README.md`.

## How to run

```
python src/data_loading.py      # EDA: class counts + sample grid
python src/train_baseline.py    # baseline CNN (from scratch)
python src/train_augmented.py   # baseline CNN + data augmentation
python src/train_resnet.py      # ResNet-18 transfer learning
python src/compare.py           # build the comparison table
python src/error_analysis.py    # error analysis of the best model (ResNet-18)
```

Results (curves, confusion matrices, comparison table, error grid) are saved to `results/`.

## External libraries used

- PyTorch / torchvision (models, transforms, pretrained ResNet-18)
- scikit-learn (precision/recall/F1, confusion matrix)
- matplotlib, NumPy, Pillow

## AI / external-tool disclosure

AI assistance was used to scaffold the repository and explain concepts; all code is
understood and adapted by the author. See the disclosure note in `proposal.md` and `final-report.md`.
