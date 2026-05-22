# Week 4 Report

## What was completed
- Error analysis of the best model (ResNet-18): `src/error_analysis.py` finds all
  misclassified test images, prints the top confused class pairs, and saves a grid of
  example mistakes (`results/error_analysis.png`).
- Wrote the final report (`final-report.md`) covering all required sections.
- Prepared the presentation outline (`presentation_outline.md`).

## Important commits / files
- `src/error_analysis.py`
- `final-report.md`
- `presentation_outline.md`
- `results/error_analysis.png`

## Experiments / analysis
- Identified that ResNet-18's remaining errors concentrate on glacier ↔ mountain
  (glacier recall 0.841). Easiest class: forest (F1 0.988).
- Confirmed the honest augmentation finding from Week 3.

## Results so far (fill from error_analysis.py output)
- Total ResNet-18 errors on test set: 221 / 3000 (~7.4%).
- Top confused pairs: glacier → mountain (66), street → buildings (46), mountain → glacier (40).
- Two confusion clusters: glacier <-> mountain (106 errors total) and street <-> buildings (66 errors total).

## Final deliverables checklist
- [ ] proposal.md (name filled)
- [ ] src/ code (data_loading, model, engine, train_baseline, train_augmented, train_resnet, compare, error_analysis)
- [ ] reports/week-01..week-04.md (real numbers filled)
- [ ] results/ figures (curves, confusion matrices, comparison table, error_analysis)
- [ ] final-report.md (name + GitHub link filled)
- [ ] presentation_outline.md (turned into slides)

## Conclusion
Project complete: baseline + two comparisons (augmentation, transfer learning),
clear best model (ResNet-18, 92.7%), full evaluation and error analysis.
