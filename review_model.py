"""Feature construction shared by training and inference."""
from collections import Counter
from datetime import datetime, timezone
import math
import numpy as np
from scipy.sparse import csr_matrix, hstack

META_COLUMNS = (
    "rating", "verified_purchase", "helpful_votes", "log_chars", "log_words",
    "log_exclamations", "log_reviewer_batch_count", "log_product_day_count",
)

def _number(value):
    try:
        if value is None or str(value).strip() == "": return np.nan
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan

def attach_history_features(records):
    """Add only within-upload reviewer and product/date counts; never persist IDs."""
    reviewer_counts = Counter(str(r.get("reviewer_id")) for r in records if r.get("reviewer_id"))
    product_day_counts = Counter(
        (str(r.get("product")), str(r.get("date"))[:10])
        for r in records if r.get("product") and r.get("date")
    )
    for row in records:
        rid = row.get("reviewer_id")
        product, date = row.get("product"), row.get("date")
        row["reviewer_batch_count"] = reviewer_counts.get(str(rid), np.nan) if rid else np.nan
        row["product_day_count"] = product_day_counts.get((str(product), str(date)[:10]), np.nan) if product and date else np.nan
    return records

def metadata_matrix(records):
    values = []
    for row in records:
        text = str(row.get("review", ""))
        words = text.split()
        verified = row.get("verified_purchase")
        if isinstance(verified, str):
            verified = verified.strip().lower() in {"true", "yes", "1", "y"}
        values.append([
            _number(row.get("rating")),
            float(verified) if isinstance(verified, bool) else _number(verified),
            np.log1p(max(0, _number(row.get("helpful_votes")))) if math.isfinite(_number(row.get("helpful_votes"))) else np.nan,
            np.log1p(len(text)), np.log1p(len(words)), np.log1p(text.count("!")),
            np.log1p(max(0, _number(row.get("reviewer_batch_count")))) if math.isfinite(_number(row.get("reviewer_batch_count"))) else np.nan,
            np.log1p(max(0, _number(row.get("product_day_count")))) if math.isfinite(_number(row.get("product_day_count"))) else np.nan,
        ])
    return np.asarray(values, dtype=np.float64)

def feature_matrix(records, bundle, fit=False):
    texts = [str(row.get("review", "")) for row in records]
    text_features = bundle["vectorizer"].fit_transform(texts) if fit else bundle["vectorizer"].transform(texts)
    numeric = metadata_matrix(records)
    if fit:
        numeric = bundle["imputer"].fit_transform(numeric)
        numeric = bundle["scaler"].fit_transform(numeric)
    else:
        numeric = bundle["scaler"].transform(bundle["imputer"].transform(numeric))
    return hstack((text_features, csr_matrix(numeric)), format="csr", dtype=np.float32)

def parse_timestamp(value):
    """Convert Amazon millisecond timestamps or ISO dates into a UTC date string."""
    if value is None or str(value).strip() == "": return None
    try:
        number = float(value)
        if number > 10_000_000_000: number /= 1000
        return datetime.fromtimestamp(number, timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        pass
    try:
        return datetime.fromisoformat(str(value).strip().replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None
