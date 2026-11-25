import io
import csv
import numpy as np
import pytest
import app as service
from review_model import feature_matrix

TEXT = "I used these headphones for two weeks. The sound is clear but the cable feels thin."

@pytest.fixture
def client():
    service.app.config["TESTING"] = True
    service.requests_window.clear()
    return service.app.test_client()

def test_inference_uses_isolation_forest(client):
    response = client.post("/api/analyze", json={"review": TEXT, "product": "Headphones"})
    assert response.status_code == 200
    result = response.json["results"][0]
    record = {"review": TEXT, "product": "Headphones"}
    features = feature_matrix([record], service.model)
    raw_score = service.model["forest"].decision_function(features)[0]
    assert result["score"] == round(float(-raw_score), 4)
    assert result["unusual"] == (service.model["forest"].predict(features)[0] == -1)
    assert result["label"] in {"Unusual pattern", "Typical pattern"}

@pytest.mark.parametrize("payload", [{}, [], {"review":""}, {"review":23}, {"review":"short"},
                                      {"review":"x"*5001}, {"review":TEXT,"product":42}, {"review":TEXT,"product":"x"*121}])
def test_invalid_single_review(client, payload):
    assert client.post("/api/analyze", json=payload).status_code == 400

def upload(client, content, name="data.csv"):
    return client.post("/api/bulk", data={"file":(io.BytesIO(content), name)}, content_type="multipart/form-data")

def test_amazon_columns_product_history_and_similarity(client):
    data = client.get("/sample.csv").data
    response = upload(client, data)
    assert response.status_code == 200
    assert response.json["summary"]["total"] == 3
    groups = response.json["summary"]["products"]
    assert len(groups) == 2
    assert groups[0]["average_rating"] is not None
    first = response.json["results"][0]
    assert first["nearest_row"] == 2
    assert first["similarity"] > .8
    assert first["reviewer_repeats"] == 2
    assert first["product_day_count"] == 2

def test_amazon_reviews_23_raw_headers(client):
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["text", "title", "parent_asin", "rating", "sort_timestamp", "user_id", "verified_purchase", "helpful_vote", "images"])
    writer.writerow(["I use it every morning and it feels sturdy after several weeks.", "Daily use", "B000000001", "5", "1634275259292", "U001", "true", "3", "[]"])
    response = upload(client, stream.getvalue().encode())
    assert response.status_code == 200
    result = response.json["results"][0]
    assert result["product"] == "B000000001"
    assert result["rating"] == 5
    assert result["verified_purchase"] is True
    assert result["helpful_votes"] == 3
    assert result["date"] == "2021-10-15"

@pytest.mark.parametrize("content,name", [(b"wrong\nx", "data.csv"), (b"text\n", "data.csv"), (b"\xff", "data.csv"),
                                           (b"text\nx", "data.txt"), (b"text,text\na,b", "data.csv"),
                                           (b'text\n"unclosed', "data.csv"), (b"text,other\na,b", "data.csv"),
                                           (b"text,rating\nvalid review text,9", "data.csv")])
def test_bad_csv(client, content, name):
    assert upload(client, content, name).status_code == 400

def test_large_upload_limits(client):
    too_many = ("text\n" + (TEXT + "\n") * (service.MAX_REVIEWS + 1)).encode()
    assert upload(client, too_many).status_code == 400
    assert service.app.config["MAX_CONTENT_LENGTH"] == 10 * 1024 * 1024

def test_health_security_and_no_store(client):
    assert client.get("/health").json["status"] == "ok"
    for path in ["/", "/analyze", "/bulk", "/dashboard", "/about"]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["Cache-Control"] == "no-store"
        assert "script-src 'self'" in response.headers["Content-Security-Policy"]

def test_rate_limit(client, monkeypatch):
    monkeypatch.setattr(service, "LIMIT", 1)
    client.post("/api/analyze", json={"review":TEXT})
    assert client.post("/api/analyze", json={"review":TEXT}).status_code == 429

def test_model_missing(client, monkeypatch):
    monkeypatch.setattr(service, "model", None)
    assert client.get("/health").status_code == 503
    assert client.post("/api/analyze", json={"review":TEXT}).status_code == 503

def test_user_content_is_not_inserted_as_html(client):
    product = "<img src=x onerror=alert(1)>"
    response = client.post("/api/analyze", json={"review":TEXT, "product":product})
    assert response.json["results"][0]["product"] == product
    js = (service.ROOT / "static/app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in js and "textContent" in js
