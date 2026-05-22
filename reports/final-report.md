# Final Report — Natural Scene Image Classification

**Course:** Deep Learning, Narxoz University (Spring 2026)
**Target level:** C+
**Author:** <Myrzabek Assanali>
**Repository:** <https://github.com/assanalimyrzabek-hub/deep-learning-final-project>

---

## 1. Project Title
CNN from Scratch vs. Transfer Learning for Natural Scene Image Classification.

## 2. Problem Statement
The task is to classify a photo of a natural scene into one of 6 categories:
*buildings, forest, glacier, mountain, sea, street*. Scene classification is useful
for photo organization, search and tagging. The project also studies a concrete
question: how much does a pretrained network (transfer learning) help compared to a
CNN trained from scratch, and does data augmentation improve a small CNN?

## 3. Dataset
- **Name / source:** Intel Image Classification, Kaggle
  (https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
- **Size:** ~14,000 training images, ~3,000 test images, 6 classes.
- **Format:** RGB JPEG images, folder-per-class.
- The raw dataset is not committed to the repository (only download instructions in `data/README.md`).

## 4. Data Preprocessing
- Images resized to a fixed size: 150×150 for the baseline CNN, 224×224 for ResNet-18.
- Converted to tensors with values scaled to [0, 1].
- For ResNet-18, images were normalized with the ImageNet mean/std, because the
  network was pretrained on ImageNet and expects the same input statistics.
- **Split:** the official training folder was split 80% train / 20% validation
  (fixed random seed = 42 for reproducibility). The official test folder (~3,000 images)
  was kept untouched and used only for the final evaluation.
- **Augmentation (one experiment only, train set):** random horizontal flip,
  rotation up to ±15°, and small brightness/contrast jitter.

## 5. Models
- **Baseline CNN (from scratch):** 3 convolutional blocks (Conv → ReLU → MaxPool),
  channels 3→32→64→128, adaptive average pooling to 4×4, then two fully-connected
  layers (2048→256→6) with dropout 0.5.
- **ResNet-18 (transfer learning):** a network pretrained on ImageNet; the final
  classification layer was replaced with a new `Linear(512 → 6)` and the network was
  fine-tuned on our data.

## 6. Training Setup
- **Loss:** Cross-Entropy. **Optimizer:** Adam.
- Baseline CNN (with and without augmentation): 10 epochs, learning rate 1e-3, batch size 32.
- ResNet-18: 5 epochs, learning rate 1e-4, batch size 32.
- Hardware: Google Colab GPU (Tesla T4).

## 7. Evaluation Metrics
Accuracy, per-class precision / recall / F1-score, macro-averaged F1, and a confusion matrix.

## 8. Results
| Model | Test accuracy | Macro F1 |
| ----- | ------------- | -------- |
| ResNet-18 (transfer learning)    | **92.7%** | 0.928 |
| Baseline CNN (no augmentation)   | 84.6% | 0.849 |
| Baseline CNN + augmentation      | 81.8% | 0.820 |

Training curves are in `results/*_curves.png`; confusion matrices in `results/*_confusion.png`.

## 9. Error Analysis
- **Best model (ResNet-18):** 221 of 3000 test images are misclassified (~7.4%).
  Errors form two clear clusters: glacier <-> mountain (106 errors: glacier->mountain 66,
  mountain->glacier 40) and street <-> buildings (66 errors). These are semantically
  similar scenes (snow/rock/sky; or buildings seen from streets), not random mistakes.
  The easiest class is forest (F1 0.992). Example mistakes: `results/error_analysis.png`.
- **Baseline vs ResNet:** the baseline CNN heavily over-predicted "mountain"
  (precision only 0.69–0.73) and missed many "sea" images (recall ~0.73). Transfer
  learning fixed most of this (sea recall rose to 0.99, glacier F1 from ~0.79 to 0.88).
- **Augmentation result (honest negative result):** augmentation slightly *lowered*
  test accuracy (84.6% → 81.8%). The baseline's overfitting was already mild, so the
  extra regularization did not help; with the same 10-epoch budget the augmented model
  underfit (train accuracy dropped from 0.841 to 0.815). The train/validation gap did
  shrink, confirming augmentation reduces overfitting — it just traded a little accuracy
  for that in this short training run.

## 10. Limitations
- Few epochs for ResNet (5) and the baseline (10) due to GPU time; longer training or
  early stopping could change the augmentation result.
- Only one pretrained architecture (ResNet-18) was tested.
- The dataset is relatively clean; performance on noisy real-world photos may be lower.

## 11. Conclusion
Transfer learning with a pretrained ResNet-18 clearly outperformed a CNN trained from
scratch (92.7% vs 84.6% test accuracy) and required fewer epochs, confirming the value
of reusing ImageNet features for a medium-sized dataset. Data augmentation did not
improve the small CNN here because its overfitting was mild and training was short.
The main remaining difficulty for all models is distinguishing visually similar scenes
(glacier vs mountain).

## 12. References
- Intel Image Classification dataset (Kaggle).
- He et al., "Deep Residual Learning for Image Recognition" (ResNet), 2015.
- PyTorch / torchvision documentation (pretrained models, transforms).
- scikit-learn documentation (classification metrics).

---

## AI / External Tools Disclosure
An AI assistant (LLM) was used to help scaffold the repository, draft documentation,
and explain deep-learning concepts. All code was reviewed, run, and understood by the
author, who can explain every part. Library components used as-is: PyTorch/torchvision
(including the pretrained ResNet-18) and scikit-learn metrics.
