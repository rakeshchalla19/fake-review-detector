"""ReviewLens: stateless Flask API for Amazon-trained review-pattern analysis."""
from collections import Counter, deque
from pathlib import Path
from threading import Lock
import csv, io, json, math, os, re, time
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template, Response
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.exceptions import HTTPException
from review_model import attach_history_features, feature_matrix, parse_timestamp

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_REVIEWS = 5000
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
MODEL_PATH = ROOT / "models" / "model.joblib"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
report = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8")) if (ROOT / "reports" / "metrics.json").exists() else {}
lock = Lock()
requests_window = deque()
LIMIT = int(os.getenv("REQUESTS_PER_MINUTE", "30"))

def validate_single(text, product=""):
    if not isinstance(text, str): raise ValueError("Review must be text.")
    text = text.strip()
    if not 30 <= len(text) <= 5000 or len(re.findall(r"[A-Za-z]+", text)) < 6:
        raise ValueError("Use an English review with 30–5,000 characters and at least 6 words.")
    if any(ord(c) < 32 and c not in "\n\r\t" for c in text): raise ValueError("Review contains unsupported control characters.")
    if not isinstance(product, str) or len(product) > 120: raise ValueError("Product name must be text, up to 120 characters.")
    return text, product.strip() or "Ungrouped"

def indicators(text):
    words = re.findall(r"\b\w+\b", text.lower())
    trigrams = [" ".join(words[i:i+3]) for i in range(max(0, len(words)-2))]
    repeats = [s for s, n in Counter(trigrams).items() if n >= 3]
    promo = [s for s in ["buy now", "must buy", "best ever", "highly recommend", "perfect product", "amazing product", "limited offer"] if s in text.lower()]
    return {"repeated_phrases": repeats[:5], "promotional_phrases": promo, "exclamation_marks": text.count("!")}

def explain_pattern(row, unusual, signals, nearest_row, similarity):
    """Return a cautious model summary plus observable context from this upload."""
    if unusual:
        summary = "The model marked this review as unusual compared with its training baseline. This is not a fake-review verdict."
    else:
        summary = "The model found no unusual pattern compared with its training baseline. This does not verify the review as genuine."

    context = []
    if nearest_row is not None and similarity is not None and similarity >= 0.75:
        context.append(f"Wording is similar to review row {nearest_row} (text similarity {similarity:.2f}).")
    if signals["repeated_phrases"]:
        phrases = ", ".join(f"‘{phrase}’" for phrase in signals["repeated_phrases"][:2])
        context.append(f"Phrase repeated 3 or more times in this review: {phrases}.")
    if signals["promotional_phrases"]:
        context.append("Promotional phrase match: " + ", ".join(signals["promotional_phrases"][:3]) + ".")
    reviewer_count = row.get("reviewer_batch_count")
    if reviewer_count is not None and reviewer_count > 1:
        context.append(f"This reviewer ID appears {reviewer_count} times in the uploaded file.")
    product_day_count = row.get("product_day_count")
    if product_day_count is not None and product_day_count > 1:
        context.append(f"{product_day_count} reviews for this product share this date in the uploaded file.")
    if not context:
        context.append("No additional contextual signals were detected in the supplied rows.")
    return summary, context

def _optional_rating(value, row_number):
    if value is None or not str(value).strip(): return None
    try: number = float(value)
    except (TypeError, ValueError): raise ValueError(f"CSV row {row_number}: rating must be between 1 and 5.")
    if not math.isfinite(number) or not 1 <= number <= 5: raise ValueError(f"CSV row {row_number}: rating must be between 1 and 5.")
    return number

def _optional_bool(value, row_number):
    if value is None or not str(value).strip(): return None
    value = str(value).strip().lower()
    if value in {"true", "yes", "1", "y"}: return True
    if value in {"false", "no", "0", "n"}: return False
    raise ValueError(f"CSV row {row_number}: verified_purchase must be true or false.")

def _bulk_record(row, row_number):
    row = {(key or "").strip().lower(): value for key, value in row.items() if key is not None}
    text = row.get("review") if "review" in row else row.get("text", "")
    title = str(row.get("title") or "").strip()
    text = str(text or "").strip()
    if title and text: text = f"{title}. {text}"
    if not 8 <= len(text) <= 10000 or not re.search(r"[A-Za-z]{2,}", text):
        raise ValueError(f"CSV row {row_number}: review text must contain 8–10,000 characters.")
    product = str(row.get("product") or row.get("parent_asin") or row.get("asin") or "Ungrouped").strip()
    if len(product) > 120: raise ValueError(f"CSV row {row_number}: product/ASIN must be at most 120 characters.")
    reviewer = str(row.get("reviewer_id") or row.get("user_id") or "").strip()
    if len(reviewer) > 200: raise ValueError(f"CSV row {row_number}: reviewer_id must be at most 200 characters.")
    raw_date = row.get("date") or row.get("timestamp") or row.get("sort_timestamp")
    date = parse_timestamp(raw_date)
    if raw_date and not date: raise ValueError(f"CSV row {row_number}: date/timestamp must be an ISO date or Unix timestamp.")
    helpful = row.get("helpful_votes") if "helpful_votes" in row else row.get("helpful_vote")
    if helpful is not None and str(helpful).strip():
        try: helpful = int(float(helpful))
        except (TypeError, ValueError): raise ValueError(f"CSV row {row_number}: helpful_votes must be a non-negative number.")
        if helpful < 0: raise ValueError(f"CSV row {row_number}: helpful_votes must be a non-negative number.")
    else: helpful = None
    return {"review": text, "product": product or "Ungrouped", "rating": _optional_rating(row.get("rating"), row_number),
            "date": date, "reviewer_id": reviewer or None, "verified_purchase": _optional_bool(row.get("verified_purchase"), row_number),
            "helpful_votes": helpful}

def analyze(records):
    if model is None: raise RuntimeError("Review-pattern model unavailable. Run the documented training command.")
    records = attach_history_features(records)
    matrix = feature_matrix(records, model)
    raw_scores = model["forest"].decision_function(matrix)
    unusual = model["forest"].predict(matrix) == -1
    texts = [r["review"] for r in records]
    nearest_rows = nearest_scores = None
    if len(records) > 1:
        try:
            similarity = TfidfVectorizer(ngram_range=(1, 2), dtype=np.float32).fit_transform(texts)
            nearest_rows = np.empty(len(records), dtype=np.int32)
            nearest_scores = np.empty(len(records), dtype=np.float32)
            for start in range(0, len(records), 64):
                end = min(start + 64, len(records))
                block = (similarity[start:end] @ similarity.T).toarray()
                block[np.arange(end-start), np.arange(start, end)] = -1
                nearest_rows[start:end] = block.argmax(axis=1)
                nearest_scores[start:end] = block[np.arange(end-start), nearest_rows[start:end]]
        except ValueError: pass
    results = []
    for i, row in enumerate(records):
        nearest = int(nearest_rows[i]) if nearest_rows is not None else None
        nearest_row = nearest+1 if nearest is not None else None
        similarity_score = round(float(nearest_scores[i]), 4) if nearest is not None else None
        row_indicators = indicators(row["review"])
        explanation, context_signals = explain_pattern(
            row, bool(unusual[i]), row_indicators, nearest_row, similarity_score
        )
        results.append({"row": i+1, "review": row["review"], "product": row["product"],
                        "label": "Unusual pattern" if unusual[i] else "Typical pattern", "unusual": bool(unusual[i]),
                        "score": round(float(-raw_scores[i]), 4), "rating": row.get("rating"), "date": row.get("date"),
                        "verified_purchase": row.get("verified_purchase"), "helpful_votes": row.get("helpful_votes"),
                        "reviewer_repeats": None if not row.get("reviewer_id") else row.get("reviewer_batch_count"),
                        "product_day_count": None if not row.get("date") else row.get("product_day_count"),
                        "indicators": row_indicators, "nearest_row": nearest_row,
                        "similarity": similarity_score, "explanation": explanation,
                        "context_signals": context_signals})
    grouped = {}
    for result in results: grouped.setdefault(result["product"], []).append(result)
    groups = []
    for product, group in grouped.items():
        ratings = [r["rating"] for r in group if r["rating"] is not None]
        verified = [r["verified_purchase"] for r in group if r["verified_purchase"] is not None]
        groups.append({"product": product, "total": len(group), "unusual": sum(r["unusual"] for r in group),
                       "proportion": sum(r["unusual"] for r in group)/len(group),
                       "average_rating": round(sum(ratings)/len(ratings), 2) if ratings else None,
                       "verified_share": round(sum(verified)/len(verified), 4) if verified else None})
    return {"results": results, "summary": {"total": len(results), "unusual": sum(r["unusual"] for r in results), "products": groups},
            "score_meaning": "Uncalibrated anomaly score: higher values are more unusual relative to the training sample. It is not a probability or proof of fraud.",
            "limitation": report.get("label_note", "Unlabeled review data cannot establish fake or genuine ground truth.")}

@app.before_request
def limit_requests():
    if request.path.startswith("/api/") and request.method == "POST":
        now = time.monotonic()
        with lock:
            while requests_window and requests_window[0] < now-60: requests_window.popleft()
            if len(requests_window) >= LIMIT: return jsonify(error="Service is busy. Please retry in one minute."), 429, {"Retry-After": "60"}
            requests_window.append(now)

@app.after_request
def secure(response):
    response.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
                             "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",
                             "Cache-Control": "no-store"})
    return response

@app.errorhandler(Exception)
def error(e):
    if isinstance(e, HTTPException):
        message = f"Upload exceeds the {MAX_UPLOAD_BYTES//(1024*1024)} MB limit." if e.code == 413 else e.description
        return jsonify(error=message), e.code
    if isinstance(e, ValueError): return jsonify(error=str(e)), 400
    return jsonify(error="Analysis is temporarily unavailable. Please retry later."), 503

@app.get("/")
@app.get("/analyze")
@app.get("/bulk")
@app.get("/dashboard")
@app.get("/about")
def page(): return render_template("index.html")

@app.get("/health")
def health(): return jsonify(status="ok" if model is not None else "unavailable"), 200 if model is not None else 503

@app.get("/api/model")
def model_info(): return jsonify(report)

@app.post("/api/analyze")
def single():
    data = request.get_json(silent=True)
    if not isinstance(data, dict): raise ValueError("Send a JSON object with a review field.")
    text, product = validate_single(data.get("review"), data.get("product", ""))
    return jsonify(analyze([{"review": text, "product": product}]))

@app.post("/api/bulk")
def bulk():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"): raise ValueError("Choose a UTF-8 .csv file.")
    try: content = f.read().decode("utf-8-sig")
    except UnicodeDecodeError: raise ValueError("Save your CSV as UTF-8 and try again.")
    try:
        reader = csv.DictReader(io.StringIO(content), strict=True)
        fields = [str(name or "").strip().lower() for name in (reader.fieldnames or [])]
        allowed = {"review", "text", "title", "product", "asin", "parent_asin", "rating", "date", "timestamp",
                   "reviewer_id", "user_id", "verified_purchase", "helpful_votes", "helpful_vote", "sort_timestamp", "images"}
        if not fields or not ({"review", "text"} & set(fields)) or len(set(fields)) != len(fields):
            raise ValueError("CSV needs a unique review or text column; product/asin and review history fields are optional.")
        if set(fields)-allowed: raise ValueError("Allowed fields: review/text, title, product/asin/parent_asin, rating, date/timestamp, reviewer_id/user_id, verified_purchase, helpful_votes, images.")
        records = []
        for row_number, row in enumerate(reader, 2):
            if row_number > MAX_REVIEWS+1: raise ValueError(f"Maximum {MAX_REVIEWS:,} reviews per upload.")
            if None in row or any(value is None for value in row.values()): raise ValueError(f"CSV row {row_number} has an incorrect number of columns.")
            records.append(_bulk_record(row, row_number))
        if not records: raise ValueError("CSV contains no reviews.")
    except csv.Error: raise ValueError("Malformed CSV. Check quotation marks and delimiters.")
    return jsonify(analyze(records))

@app.get("/sample.csv")
def sample():
    stream = io.StringIO(); writer = csv.writer(stream)
    writer.writerow(["review", "product", "rating", "date", "reviewer_id", "verified_purchase", "helpful_votes"])
    writer.writerow(["I used these headphones for two weeks. The sound is clear but the cable feels thin.", "Headphones", "4", "2024-03-16", "reviewer-001", "true", "2"])
    writer.writerow(["The sound is clear but the cable feels thin. I used these headphones for two weeks.", "Headphones", "4", "2024-03-16", "reviewer-001", "true", "1"])
    writer.writerow(["This bottle fits in my bag and has not leaked during my commute. The lid is a little stiff.", "Water bottle", "3", "2024-02-10", "reviewer-002", "false", "0"])
    return Response(stream.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=sample-reviews.csv"})

if __name__ == "__main__": app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
