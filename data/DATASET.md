# Amazon Review Baseline Data

The active training source is the [Amazon Reviews ’23 dataset](https://amazon-reviews-2023.github.io/) maintained by McAuley Lab at UCSD. This project uses the `All_Beauty` review category.

The source includes review text, rating, product ASIN / parent ASIN, reviewer ID, timestamp, verified-purchase status, and helpful votes. The dataset describes 701,500 All_Beauty reviews. Training filters empty/very short text and takes a deterministic 50,000-review reservoir sample from the usable rows; `reports/metrics.json` records the exact sample size and source file checksum.

**The source has no fake/genuine labels.** The model is unsupervised and uses Isolation Forest to rank statistical outliers. An outlier score is not a fake-review probability and does not establish deception. No supervised accuracy, precision, recall, or F1 score is available for this data.

Training steps:

1. Download the official category file with `python scripts/download_data.py`.
2. Stream the JSONL/GZIP data and make a seeded reservoir sample.
3. Build word/phrase TF-IDF and structured review-history features.
4. Fit an Isolation Forest with a fixed 5% shortlist setting.
5. Save the model bundle to `models/model.joblib` and training provenance to `reports/metrics.json`.

The raw downloaded corpus is excluded from Git. Source data is not included in the project archive. Check the dataset provider's current terms for use outside this research project.

Citation: Hou, Y. et al. (2024). *Bridging Language and Items for Retrieval and Recommendation*. Amazon Reviews ’23. https://amazon-reviews-2023.github.io/ .
