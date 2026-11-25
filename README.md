# ReviewLens — Product Review Pattern Analysis

A Flask and scikit-learn project for exploring unusual patterns in product-review history. It supports a single review or a CSV upload with up to 5,000 rows and 10 MB, product-level summaries, paged review details, and CSV export.

**What the model does:** an unsupervised Isolation Forest ranks reviews that differ from an Amazon review baseline. It does not label reviews as fake or genuine and its score is not a probability of fraud. The source corpus has no fake/genuine ground truth.

## Live application

Website: https://fake-review-detector-fjzd.onrender.com/

## Run locally

Use Python 3.12 and the pinned dependencies:

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The included model is loaded at startup. No training data or API keys are needed to run inference.

## Train the model

Training uses Amazon Reviews ’23, a large research corpus with product IDs, text, ratings, reviewer IDs, timestamps, verified-purchase flags, and helpful votes. It does not include fake/genuine labels. The project downloads the All_Beauty category, then takes a deterministic 50,000-row reservoir sample for training. Raw data is ignored by Git and is not redistributed here.

```bash
python scripts/download_data.py
python train.py
```

The model combines TF-IDF word and phrase features with rating, verified-purchase, helpfulness, text length, reviewer repetition, and product/date activity features. Isolation Forest uses a 5% shortlist threshold. This setting controls how many reviews are flagged as unusual; it does not mean 5% are fake.

Dataset source and citation: [Amazon Reviews ’23](https://amazon-reviews-2023.github.io/), Hou et al., *Bridging Language and Items for Retrieval and Recommendation* (2024). The raw review data remains in the source project; verify its current terms for your intended use.

## Bulk CSV format

Required column: `review` or the Amazon dataset’s `text`.

Optional columns are accepted directly from Amazon Reviews ’23: `title`, `asin`, `parent_asin`, `rating`, `timestamp`, `user_id`, `verified_purchase`, and `helpful_vote`. Generic aliases are also supported: `product`, `date`, `reviewer_id`, and `helpful_votes`.

The app accepts up to 5,000 rows / 10 MB. Product and reviewer IDs are used only to calculate counts within the uploaded file. Uploads and results are processed in memory and are not persisted by this application. Amazon product URLs are not fetched.

Download a sample CSV from `/sample.csv`.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Model availability |
| `/api/model` | GET | Dataset and training-method details |
| `/api/analyze` | POST | Analyze one English review |
| `/api/bulk` | POST | Analyze a CSV upload |
| `/sample.csv` | GET | Download a sample file |

## Hosting

The Render Blueprint uses Python 3.12, Gunicorn, a single worker, a 180-second request timeout, and the tracked model artifact. Training is run offline and is not part of the deployment build. The free service may sleep when idle. Check the Render dashboard after pushing changes to confirm that the deploy is live.

## Limitations

- Amazon Reviews ’23 has no verified fake/genuine labels, so an anomaly score cannot establish deception or authenticity.
- Training uses one category and a 50,000-row sample; broad category generalization has not been validated.
- Reviewer and product history features in an uploaded file are limited to that file. They do not access Amazon’s full live history.
- Reviews are scored against a population baseline, so unusual but legitimate writing can be flagged and coordinated deception can look ordinary.
- Product-link retrieval and marketplace scraping are not included. Use only data you have permission to process.

Never use this demo alone to accuse reviewers or make consequential decisions.
