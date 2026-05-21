# Project Proposal

**Course:** Deep Learning, Narxoz University (Spring 2026)
**Target level:** C+ (baseline + improvements / comparison)
**Author:** <ВАШЕ ИМЯ>

---

## 1. Project Title

**CNN from Scratch vs. Transfer Learning for Natural Scene Image Classification**

## 2. Problem Statement

1. **What problem are we solving?** Given a photo of a natural scene, automatically
   predict which of 6 categories it belongs to: *buildings, forest, glacier,
   mountain, sea, street*.
2. **Why is it useful?** Scene classification is a building block for photo
   organization, geotagging, content moderation and search. It is also a clean
   setting to study *how much* a pretrained network helps compared to a model
   trained from scratch on a limited dataset.
3. **What does the model predict?** A single class label (one of 6) for each input
   image, plus a probability for each class.

## 3. Dataset

- **Name:** Intel Image Classification
- **Source:** https://www.kaggle.com/datasets/puneet6060/intel-image-classification
- **Size:** ~14,000 training/validation images, ~3,000 test images.
- **Input:** RGB images, originally 150×150 px (resized in preprocessing).
- **Target labels:** 6 classes — buildings, forest, glacier, mountain, sea, street.
- **Format:** folder-per-class (`seg_train/`, `seg_test/`), JPEG images.
- **License / usage:** public Kaggle dataset, used for educational purposes; source cited above.

## 4. Planned Method

- **Baseline:** a small Convolutional Neural Network (CNN) built from scratch
  (3 conv blocks + 2 fully-connected layers).
- **Deep learning model (improvement 1):** transfer learning — a pretrained
  **ResNet-18** (ImageNet weights), with the final layer replaced for 6 classes and fine-tuned.
- **Improvement 2 (comparison):** effect of **data augmentation** — train the baseline
  CNN with vs. without augmentation (random flips, rotation, color jitter) and compare.
- **Loss function:** Cross-Entropy Loss.
- **Evaluation metrics:** accuracy, per-class precision / recall / F1-score,
  confusion matrix, and training/validation loss & accuracy curves.
- **Split plan:** use the official `seg_train` set, split into **80% train / 20% validation**;
  keep `seg_test` as the **untouched test set** used only for the final evaluation.

## 5. Expected Challenges

1. **Class confusion:** glacier vs. mountain and buildings vs. street look visually
   similar and are likely to be mixed up.
2. **Overfitting:** the from-scratch CNN may memorize the training set; augmentation
   and validation monitoring are planned to detect and reduce this.
3. **Training time / hardware:** training on ~14k images needs a GPU; plan to use
   Google Colab's free GPU.
4. **Fair comparison:** all three setups must use the same split, image size, and
   number of epochs so the comparison is honest.

## 6. Weekly Plan

| Week   | Planned Work                                                          | Expected Output                          |
| ------ | --------------------------------------------------------------------- | ---------------------------------------- |
| Week 1 | Topic + dataset selection, repository setup, exploratory data analysis | Proposal, README, dataset summary (EDA)  |
| Week 2 | Preprocessing, train/val/test split, baseline CNN from scratch         | Baseline results + Week 2 report         |
| Week 3 | Transfer learning (ResNet-18) + augmentation experiments               | Model results, plots, comparison table   |
| Week 4 | Error analysis, final evaluation, final report and presentation        | Final code, final report, slides         |

---

## AI / External Tools Disclosure

In line with the course rules, external help is disclosed here and will be kept
up to date: AI assistance (an LLM) was used to help structure the repository,
draft this proposal, and explain deep-learning concepts. All code is reviewed,
understood, and re-written/adapted by the author, who can explain every part
during the defense. Library code (PyTorch, torchvision pretrained ResNet-18,
scikit-learn metrics) is used as-is and cited in the README.
