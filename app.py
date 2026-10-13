"""ReviewLens: stateless Flask API and accessible server-rendered shell."""
from pathlib import Path
from collections import Counter, deque
from threading import Lock
import csv, io, json, os, re, time
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template, Response
from langdetect import detect_langs, DetectorFactory, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from werkzeug.exceptions import HTTPException
DetectorFactory.seed=42
ROOT=Path(__file__).resolve().parent
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=1024*1024
MODEL_PATH=ROOT/'models/model.joblib'
model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
report=json.loads((ROOT/'reports/metrics.json').read_text()) if (ROOT/'reports/metrics.json').exists() else {}
lock=Lock(); requests_window=deque()
# Single-worker deployment: a global cap avoids trusting spoofable proxy IP headers.
LIMIT=int(os.getenv('REQUESTS_PER_MINUTE','30'))

def validate(text,product=''):
    if not isinstance(text,str): raise ValueError('review must be text.')
    text=text.strip()
    if not 30<=len(text)<=5000 or len(re.findall(r'[A-Za-z]+',text))<6:
        raise ValueError('Use an English review with 30–5,000 characters and at least 6 words.')
    if any(ord(c)<32 and c not in '\n\r\t' for c in text): raise ValueError('Review contains unsupported control characters.')
    if not isinstance(product,str) or len(product)>120: raise ValueError('Product name must be text, up to 120 characters.')
    try:
        languages=detect_langs(text)
        if not languages or languages[0].lang!='en' or languages[0].prob<.80:
            raise ValueError('English reviews only. Language could not be reliably identified; try a longer English review.')
    except LangDetectException: raise ValueError('Could not identify the review language.')
    return text,product.strip() or 'Ungrouped'

def indicators(text):
    words=re.findall(r'\b\w+\b',text.lower()); trigrams=[' '.join(words[i:i+3]) for i in range(len(words)-2)]
    repeats=[s for s,n in Counter(trigrams).items() if n>=3]
    promo=[s for s in ['buy now','must buy','best ever','highly recommend','perfect product','amazing product','limited offer'] if s in text.lower()]
    return {'repeated_phrases':repeats[:5],'promotional_phrases':promo,'exclamation_marks':text.count('!')}

def analyze(rows):
    if model is None: raise RuntimeError('Model unavailable. Run the documented training command.')
    texts=[r[0] for r in rows]; margins=model.decision_function(texts)
    x=model.named_steps['tfidf'].transform(texts)
    names=model.named_steps['tfidf'].get_feature_names_out(); coef=model.named_steps['classifier'].coef_[0]
    # Similarity is request-local and separate from classifier features/explanations.
    sims=None
    if len(rows)>1:
        try:
            sx=TfidfVectorizer(ngram_range=(1,2)).fit_transform(texts); sims=(sx@sx.T).toarray(); np.fill_diagonal(sims,-1)
        except ValueError: pass
    results=[]
    for i,((text,product),margin) in enumerate(zip(rows,margins)):
        features=x.getrow(i); values=features.data*coef[features.indices]
        order=np.argsort(np.abs(values))[::-1][:6]
        contributions=[{'term':str(names[features.indices[j]]),'contribution':round(float(values[j]),4)} for j in order]
        nearest=int(np.argmax(sims[i])) if sims is not None else None
        results.append(dict(row=i+1,review=text,product=product,label='Potentially Fake' if margin>=0 else 'Likely Genuine',flagged=bool(margin>=0),score=round(float(margin),4),indicators=indicators(text),top_terms=contributions,intercept=round(float(model.named_steps['classifier'].intercept_[0]),4),nearest_row=nearest+1 if nearest is not None else None,similarity=round(float(sims[i,nearest]),4) if nearest is not None else None))
    groups=[]
    for p in dict.fromkeys(r['product'] for r in results):
        group=[r for r in results if r['product']==p]; flagged=sum(r['flagged'] for r in group)
        groups.append(dict(product=p,total=len(group),flagged=flagged,proportion=flagged/len(group)))
    return dict(results=results,summary=dict(total=len(results),flagged=sum(r['flagged'] for r in results),products=groups),score_meaning='Uncalibrated margin: positive favors CG, negative favors OR. Not a probability or confidence percentage.',limitation='CG means GPT-2-generated; OR means original, presumed human reviews. Predictions do not prove truthfulness or deception.')

@app.before_request
def limit_requests():
    if request.path.startswith('/api/') and request.method=='POST':
        now=time.monotonic()
        with lock:
            while requests_window and requests_window[0]<now-60: requests_window.popleft()
            if len(requests_window)>=LIMIT: return jsonify(error='Service is busy. Please retry in one minute.'),429,{'Retry-After':'60'}
            requests_window.append(now)

@app.after_request
def secure(response):
    response.headers.update({'X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY','Referrer-Policy':'no-referrer','Content-Security-Policy':"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'",'Cache-Control':'no-store'})
    return response

@app.errorhandler(Exception)
def error(e):
    if isinstance(e,HTTPException): return jsonify(error='Upload exceeds the 1 MB limit.' if e.code==413 else e.description),e.code
    if isinstance(e,ValueError): return jsonify(error=str(e)),400
    # Do not log submitted text or request bodies.
    return jsonify(error='Analysis is temporarily unavailable. Please retry later.'),503

@app.get('/')
@app.get('/analyze')
@app.get('/bulk')
@app.get('/dashboard')
@app.get('/about')
def page(): return render_template('index.html')
@app.get('/health')
def health(): return jsonify(status='ok' if model is not None else 'unavailable'),200 if model is not None else 503
@app.get('/api/model')
def model_info(): return jsonify(report)
@app.post('/api/analyze')
def single():
    data=request.get_json(silent=True)
    if not isinstance(data,dict): raise ValueError('Send a JSON object with a review field.')
    return jsonify(analyze([validate(data.get('review'),data.get('product',''))]))
@app.post('/api/bulk')
def bulk():
    f=request.files.get('file')
    if not f or not f.filename.lower().endswith('.csv'): raise ValueError('Choose a UTF-8 .csv file.')
    try: content=f.read().decode('utf-8-sig')
    except UnicodeDecodeError: raise ValueError('Save your CSV as UTF-8 and try again.')
    try:
        reader=csv.DictReader(io.StringIO(content),strict=True)
        if not reader.fieldnames or 'review' not in reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames): raise ValueError('CSV needs a unique review column; product is optional.')
        if set(reader.fieldnames)-{'review','product'}: raise ValueError('Allowed CSV columns: review, product.')
        rows=[]
        for n,row in enumerate(reader,2):
            if n>201: raise ValueError('Maximum 200 reviews per upload.')
            if None in row or any(v is None for v in row.values()): raise ValueError(f'CSV row {n} has an incorrect number of columns.')
            try: rows.append(validate(row['review'],row.get('product','')))
            except ValueError as e: raise ValueError(f'CSV row {n}: {e}')
        if not rows: raise ValueError('CSV contains no reviews.')
    except csv.Error: raise ValueError('Malformed CSV. Check quotation marks and delimiters.')
    return jsonify(analyze(rows))
@app.get('/sample.csv')
def sample():
    s=io.StringIO(); w=csv.writer(s); w.writerow(['review','product'])
    w.writerow(['I used these headphones for two weeks. The sound is clear but the cable feels thin.','Headphones'])
    w.writerow(['The sound is clear but the cable feels thin. I used these headphones for two weeks.','Headphones'])
    w.writerow(['This bottle fits in my bag and has not leaked during my commute. The lid is a little stiff.','Water bottle'])
    return Response(s.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=sample-reviews.csv'})
if __name__=='__main__': app.run(host='127.0.0.1',port=int(os.getenv('PORT','5000')),debug=False)
