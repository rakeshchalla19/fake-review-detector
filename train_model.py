import pandas as pd, pickle, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack, csr_matrix

df = pd.read_csv("dataset.csv")

tfidf = TfidfVectorizer(max_features=500, stop_words="english")
X_text = tfidf.fit_transform(df["review"])

def feats(row):
    t = row["review"]
    return [row["rating"], row["burst_score"], row["verified"],
            t.count("!"), len(t.split()),
            sum(c.isupper() for c in t) / max(len(t), 1)]

X_meta = np.array([feats(r) for _, r in df.iterrows()])
X = hstack([X_text, csr_matrix(X_meta)])
y = df["label"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(Xtr, ytr)

pred = model.predict(Xte)
print("Accuracy:", accuracy_score(yte, pred))
print(classification_report(yte, pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
print("Saved model.pkl and tfidf.pkl")
