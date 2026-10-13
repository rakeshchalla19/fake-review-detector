# Mini-project explanation and demonstration guide

## Project title
Fake Product Review Detection — ReviewLens

## Problem statement
Online shoppers cannot easily judge whether product feedback is reliable. Reading thousands of reviews manually is impractical. This project provides a transparent review-screening tool while making the difference between automated patterns and verified deception explicit.

## Objectives
- Train and compare real machine learning classifiers on a licensed public dataset.
- Prevent obvious train/test contamination using exact and detected near-duplicate filtering.
- Provide individual and CSV inference through a working web API.
- Explain linear feature contributions and show separate heuristic indicators.
- Evaluate on held-out data and communicate limitations responsibly.

## Architecture
Browser → Flask validation → saved TF-IDF/SVM pipeline → JSON → results dashboard and CSV. Independent heuristics examine repetition, promotional phrases, punctuation and batch similarity. They do not modify the classifier's prediction. Model inference happens on the server; there is no external AI API or database.

## Methodology
Use the Salminen et al. original-versus-GPT-2 dataset (CC BY 4.0). Check the download checksum. Normalize text for deduplication, identify approximate near-duplicate candidates with MinHash, confirm 3-word-shingle Jaccard similarity, and collapse groups before splitting. Use 60% training, 20% validation and 20% test data. Fit TF-IDF only on training reviews. Compare majority baseline, Logistic Regression and Linear SVM. Choose the classifier with the highest validation F1 and evaluate on the untouched test set. Save the pipeline with joblib.

## Results
See `reports/EVALUATION.md` for the full table. The selected SVM achieved 94.67% held-out accuracy, 95.21% CG precision, 94.03% CG recall and 94.62% CG F1. The test set contained 8,063 reviews. These numbers describe GPT-2/original classification, not universal fake-review accuracy. There were 190 original reviews classified CG and 240 CG reviews classified original.

## Limitations
The dataset uses AI-generation labels, not independently verified dishonesty. Human fake reviews, modern AI models, unfamiliar categories and non-English text are not validated. Original reviews can also be misleading. Near-duplicate detection is approximate. Product/user IDs are unavailable. A benchmark score cannot justify accusations against real reviewers. The model score is an uncalibrated margin, not confidence. Free hosting may sleep.

## Future scope
Collect licensed, independently verified deception labels; evaluate on unseen products and categories; add calibrated probabilities only after separate calibration/evaluation; develop abstention for unfamiliar inputs; measure false-positive harm; evaluate multilingual and modern-generator datasets; add permitted platform APIs if available.

## Three-minute demonstration
1. **0:00–0:30 — Define the task.** “This is a review-screening mini project. Our dataset distinguishes original from GPT-2-generated text, so we cannot prove dishonesty.” Show the landing page.
2. **0:30–1:10 — Single review.** Open Analyze, choose an example, add a product name, and submit. Read the label and explain the signed margin. Expand contributions. Do not promise a specific label for a handcrafted example.
3. **1:10–1:50 — Bulk workflow.** Download the sample CSV and upload it. Show three reviews grouped into two products. Compare the first two reviews' wording similarity.
4. **1:50–2:20 — Explainability.** Distinguish model word contributions from promotional/repetition heuristics. Download results. Open the CSV locally.
5. **2:20–2:50 — Evaluation.** Open Model & data. Explain why validation selects the model and test measures its performance. Point to false positives/negatives in the confusion matrix.
6. **2:50–3:00 — Close.** Explain memory-only review handling, the scientific limits and future improvements.

Before presenting: start locally as backup, test a single review and the sample file, have the model page ready, and wake a Free Render service in advance if deployed. Never present a fabricated live URL.

## Viva questions and answers

**1. What is your project actually predicting?**
Whether a review resembles the dataset's original class or GPT-2-generated class. These are proxies, not proof of honesty.

**2. Why use TF-IDF?**
It converts text into sparse numeric features. A term gets weight based on its frequency in a document and rarity across training documents.

**3. Why include bigrams?**
Two-word phrases capture limited local context that individual words miss.

**4. Why compare a baseline?**
A majority-class classifier tells us whether the trained system improves on a trivial decision rule. Accuracy alone can look good on imbalanced data.

**5. What is a Linear SVM?**
A classifier that learns a separating hyperplane with a margin between classes. Its decision function is a signed distance-related score, not a probability.

**6. How is Logistic Regression different?**
It learns a linear log-odds model. Although it outputs probability-like values, calibration must still be checked before claiming trustworthy confidence.

**7. What is data leakage?**
When training or selection uses information that should be unavailable from held-out data. Duplicates spanning splits and fitting TF-IDF on all data are examples.

**8. How did you handle duplicates?**
Normalized exact duplicates are removed. MinHash proposes similar pairs, Jaccard confirms them, and connected groups are collapsed before splitting. Conflicting-label groups are removed. Approximate search can miss some pairs.

**9. Why use three splits?**
Training learns parameters, validation selects a model, and the untouched test set estimates performance after selection.

**10. Define precision and recall.**
CG precision = correctly flagged CG reviews / all flagged reviews. CG recall = correctly flagged CG reviews / all actual CG reviews.

**11. What is F1?**
The harmonic mean of precision and recall: 2PR/(P+R). It balances the two, but does not measure every real-world cost.

**12. What does the confusion matrix show?**
For the SVM, rows are actual OR/CG and columns predicted OR/CG: [[3853,190],[240,3780]]. It exposes the two kinds of mistake.

**13. Is a score of +1 an 100% confidence prediction?**
No. It is an uncalibrated decision-function margin. Zero is the classification threshold.

**14. How do word explanations work?**
Multiply each TF-IDF feature by its learned linear coefficient. Add all contributions and the intercept to recover the decision margin. The UI shows only the six largest absolute contributions.

**15. Do promotional words prove a review is fake?**
No. Genuine users can use promotional wording. The indicator is a separate heuristic and does not alter the model.

**16. What does cosine similarity mean?**
It measures alignment between two TF-IDF vectors. Higher similarity suggests shared wording in the current batch, not dishonesty or common authorship.

**17. Why Flask?**
It keeps a small Python inference service and frontend in one understandable application with minimal infrastructure.

**18. How are reviews protected?**
The app does not persist them, rejects large/invalid input, uses safe text rendering, neutralizes CSV formulas, and limits requests. Infrastructure may log metadata.

**19. Can it detect all fake reviews on Amazon?**
No. There is no marketplace retrieval integration, and the training labels do not establish broad deception-detection performance.

**20. Why might performance drop in production?**
Different product categories, languages, generators, label prevalence and writing styles can shift the data distribution.

**21. What is the dataset license?**
The original OSF project explicitly links to CC BY 4.0. We retain attribution, the license reference and notes on preprocessing changes.

**22. What would you improve first?**
Collect a better-matched verified-deception dataset and run product-disjoint and external evaluation before strengthening any claims.
