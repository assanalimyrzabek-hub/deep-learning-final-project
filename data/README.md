# Natural Scene Image Classification — CNN vs. Transfer Learning

Deep Learning final project (Narxoz University, Spring 2026). Target level: **C+**.

The goal is to classify natural-scene photos into 6 classes
(*buildings, forest, glacier, mountain, sea, street*) and to compare a CNN
trained from scratch against a pretrained **ResNet-18**, plus measure the effect
of data augmentation.

## Repository structure

```
.
├── README.md            # this file
├── requirements.txt     # Python dependencies
├── proposal.md          # project proposal (Week 1)
├── data/
│   └── README.md        # how to download the dataset (data itself is NOT committed)
├── notebooks/           # Jupyter notebooks for experiments
├── src/                 # reusable Python code (data loading, models, training, eval)
├── reports/             # weekly progress reports (week-01 ... week-04)
├── results/             # saved figures, metrics tables
└── final-report.md      # final report (Week 4)
```

## Dataset

Intel Image Classification — https://www.kaggle.com/datasets/puneet6060/intel-image-classification

See `data/README.md` for download instructions. The raw dataset is **not** stored
in this repository (per course rules — no large raw data in git).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then download the dataset into `data/` as described in `data/README.md`.

## How to run

1. EDA: `python src/data_loading.py` (prints class counts and saves a sample grid to `results/`).
2. (Week 2+) baseline / transfer-learning training scripts will be added here.

## External libraries used

- PyTorch / torchvision (models, transforms, pretrained ResNet-18)
- scikit-learn (precision/recall/F1, confusion matrix)
- matplotlib, NumPy, Pillow

## AI / external-tool disclosure

AI assistance was used to scaffold the repo and explain concepts; all code is
understood and adapted by the author. See the disclosure note in `proposal.md`.
