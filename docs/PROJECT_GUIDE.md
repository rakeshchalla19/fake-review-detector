# ReviewLens project guide

## Project goal

ReviewLens helps a user inspect product-review patterns and product-level history. It is a screening aid, not a system for proving that a review is fake or genuine.

## Model and data

The active model is an Isolation Forest trained without labels on a seeded 50,000-review sample from the All_Beauty category of Amazon Reviews ’23. It combines review text features with rating, purchase verification, helpful votes, text length, reviewer repeats, and product/date activity. A 5% setting creates an unusual-pattern shortlist; it does not estimate fake-review prevalence.

Amazon Reviews ’23 contains no fake/genuine labels. Therefore the project does not report classification accuracy, precision, recall, or F1. The output is “Unusual pattern” or “Typical pattern” with an uncalibrated anomaly score. Unusual does not mean fake.

## Architecture

Browser CSV → Flask validation and in-memory processing → TF-IDF and history features → saved Isolation Forest → paged results, product summaries, and safe CSV export. The model artifact is loaded once when the web service starts. Uploaded review text and IDs are not persisted by the app.

## Train and run

```bash
python -m pip install -r requirements.txt
python scripts/download_data.py
python train.py
python app.py
```

Training uses the official [Amazon Reviews ’23 project](https://amazon-reviews-2023.github.io/). The raw dataset is downloaded separately and excluded from Git. Training provenance is written to `reports/metrics.json`.

## Bulk review format

Required: `review` or Amazon’s `text` column. Optional Amazon fields include `title`, `asin`, `parent_asin`, `rating`, `timestamp`, `user_id`, `verified_purchase`, and `helpful_vote`. The app also accepts `product`, `date`, `reviewer_id`, and `helpful_votes` aliases. Up to 5,000 rows / 10 MB are accepted. Product and reviewer counts are computed only within the uploaded file.

## Three-minute demonstration

1. Show the model/data page and explain why the dataset supports outlier ranking but not a fake/genuine verdict.
2. Download the sample CSV or use a permitted Amazon Reviews ’23 extract.
3. Upload reviews grouped by product; show unusual-pattern count, average rating, and verified-purchase share.
4. Inspect a paged result and compare its closest wording match.
5. Export results and explain that the score is not a probability or proof.

## Limits

Training uses one product category and is not validated against verified deception labels. Anomaly ranking can flag legitimate minority writing styles and miss coordinated or natural-looking deception. Product URLs are not fetched; use only review data you have permission to process. See `reports/EVALUATION.md` for evaluation status.
