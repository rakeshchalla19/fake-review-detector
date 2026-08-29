from flask import Flask, request, render_template
import pickle, numpy as np
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        review = request.form["review"]
        rating = int(request.form["rating"])
        burst = float(request.form["burst"])
        verified = int(request.form["verified"])

        X_text = tfidf.transform([review])
        meta = np.array([[rating, burst, verified, review.count("!"),
                          len(review.split()),
                          sum(c.isupper() for c in review) / max(len(review), 1)]])
        X = hstack([X_text, csr_matrix(meta)])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred]
        result = {
            "label": "FAKE REVIEW" if pred == 1 else "GENUINE REVIEW",
            "confidence": round(prob * 100, 2)
        }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
