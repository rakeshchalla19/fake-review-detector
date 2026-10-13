# Actual held-out evaluation

Source: 40,432 rows. Retained: 40,313. Removed: 119. Confirmed near-duplicate candidate pairs: 100.

Split: 24,187 train / 8,063 validation / 8,063 test. Seed 42.

Positive class: CG (GPT-2 generated). Negative: OR (original). Selected by validation CG F1: Linear SVM.

| Model | Accuracy | CG precision | CG recall | CG F1 | Validation CG F1 |
|---|---:|---:|---:|---:|---:|
| majority baseline | 50.14% | 0.00% | 0.00% | 0.00% | 0.00% |
| logistic regression | 92.89% | 94.36% | 91.19% | 92.75% | 92.74% |
| linear svm | 94.67% | 95.21% | 94.03% | 94.62% | 94.57% |

## Selected-model confusion matrix

Rows = actual, columns = predicted.

| | Predicted OR | Predicted CG |
|---|---:|---:|
| Actual OR | 3853 | 190 |
| Actual CG | 240 | 3780 |

190 original reviews were flagged and 240 generated reviews were missed. These are dataset-specific errors, not independently verified real-world false accusations.

## Per-category test results

Categories are represented in training; this is not an unseen-category evaluation.

| Category | Accuracy | CG F1 |
|---|---:|---:|
| Books_5 | 93.83% | 93.81% |
| Clothing_Shoes_and_Jewelry_5 | 94.12% | 93.84% |
| Electronics_5 | 95.29% | 94.98% |
| Home_and_Kitchen_5 | 96.33% | 96.45% |
| Kindle_Store_5 | 94.37% | 94.44% |
| Movies_and_TV_5 | 93.25% | 93.30% |
| Pet_Supplies_5 | 95.04% | 95.04% |
| Sports_and_Outdoors_5 | 95.73% | 95.65% |
| Tools_and_Home_Improvement_5 | 95.48% | 95.53% |
| Toys_and_Games_5 | 93.20% | 93.01% |

## Reproducibility and limitations

Normalized exact deduplication; MinHash LSH candidate search (128 permutations, threshold .70), verified 3-word-shingle Jaccard >= .85; connected components collapsed; conflicting-label components removed. Approximate search can miss near-duplicates. Product/user IDs unavailable; product-disjoint generalization not established.

The model retains the training-only fit; validation/test are not added afterward. No probability calibration was performed or claimed. No confidence intervals, external deception benchmark or current-generator benchmark were run. Metrics are one fixed split, not cross-validation averages.

Dataset SHA-256: `ea03422988824bc917b5f7217e3867e27b597c647d5261adfae73cd7a85e1e1a`

Versions: {'python': '3.12.14', 'sklearn': '1.8.0', 'pandas': '2.2.3'}
