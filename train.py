"""Train an unsupervised review-pattern baseline from Amazon Reviews '23."""
from collections import Counter
import gzip, hashlib, json, random
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from review_model import attach_history_features, feature_matrix, parse_timestamp

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "amazon-reviews-2023-All_Beauty.jsonl.gz"
SAMPLE_SIZE = 50_000
SEED = 42

def load_sample():
    if not SOURCE.exists():
        raise SystemExit("Amazon training data is missing. Run: python scripts/download_data.py")
    rng = random.Random(SEED)
    sample, eligible = [], 0
    with gzip.open(SOURCE, "rt", encoding="utf-8") as stream:
        for line in stream:
            try: item = json.loads(line)
            except json.JSONDecodeError: continue
            text = str(item.get("text") or "").strip()
            if len(text) < 8: continue
            title = str(item.get("title") or "").strip()
            record = {
                "review": (title + ". " + text) if title else text,
                "product": str(item.get("parent_asin") or item.get("asin") or ""),
                "reviewer_id": str(item.get("user_id") or ""),
                "date": parse_timestamp(item.get("timestamp", item.get("sort_timestamp"))),
                "rating": item.get("rating"),
                "verified_purchase": item.get("verified_purchase"),
                "helpful_votes": item.get("helpful_vote", item.get("helpful_votes")),
            }
            eligible += 1
            if len(sample) < SAMPLE_SIZE: sample.append(record)
            else:
                index = rng.randrange(eligible)
                if index < SAMPLE_SIZE: sample[index] = record
    if len(sample) < 5_000:
        raise SystemExit(f"Only {len(sample):,} usable reviews found; at least 5,000 are required.")
    return attach_history_features(sample), eligible

def main():
    records, eligible = load_sample()
    bundle = {
        "vectorizer": TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                                      max_features=30_000, sublinear_tf=True, dtype=np.float32),
        "imputer": SimpleImputer(strategy="median"),
        "scaler": StandardScaler(),
    }
    matrix = feature_matrix(records, bundle, fit=True)
    forest = IsolationForest(n_estimators=200, max_samples=512, contamination=.05,
                             random_state=SEED, n_jobs=1)
    forest.fit(matrix)
    bundle["forest"] = forest
    bundle["metadata_columns"] = ["rating", "verified purchase", "helpful votes", "review length",
                                  "word count", "exclamation count", "reviewer repeats in upload",
                                  "product/day review count"]
    bundle["training_samples"] = len(records)
    bundle["source_rows_seen"] = eligible
    bundle["source"] = "Amazon Reviews '23 — All_Beauty category, public McAuley Lab research dataset"
    bundle["algorithm"] = "Isolation Forest; unsupervised; 5% review-pattern shortlist threshold"
    model_path = ROOT / "models" / "model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    report = {
        "algorithm": bundle["algorithm"], "training_samples": len(records),
        "source_rows_seen": eligible, "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_dataset": bundle["source"], "training_scope": "All_Beauty product reviews collected through September 2023",
        "labels": None,
        "label_note": "The source data has no fake/genuine ground-truth labels. The model ranks statistical outliers; this is not a fake-review classifier or a probability of fraud.",
        "features": bundle["metadata_columns"],
        "threshold_note": "The 5% setting controls shortlist size in the training baseline; it is not an estimate that 5% of reviews are fake.",
        "random_seed": SEED,
    }
    report_path = ROOT / "reports" / "metrics.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "training_samples": len(records),
                      "source_rows_seen": eligible, "algorithm": bundle["algorithm"]}, indent=2))

if __name__ == "__main__": main()
