# Dataset: Intel Image Classification

Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

The raw images are **not** committed to git (course rule + size). Download them yourself:

## Option A — manual
1. Open the Kaggle link above and click **Download**.
2. Unzip so that this folder looks like:
   ```
   data/
   ├── seg_train/seg_train/<class>/*.jpg
   ├── seg_test/seg_test/<class>/*.jpg
   └── seg_pred/seg_pred/*.jpg
   ```

## Option B — Kaggle API
```bash
pip install kaggle                       # put kaggle.json in ~/.kaggle/
kaggle datasets download -d puneet6060/intel-image-classification -p data/
cd data && unzip intel-image-classification.zip
```

Classes (6): buildings, forest, glacier, mountain, sea, street.
