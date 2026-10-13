"""Deterministic, duplicate-aware model comparison; test set is never used to select a model."""
from pathlib import Path
import json, re, hashlib, platform
import numpy as np
import pandas as pd
import sklearn, joblib
from datasketch import MinHash, MinHashLSH
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parent
SEED=42

def normalize(s): return ' '.join(re.findall(r'\w+', str(s).lower()))
def shingles(s):
    w=s.split()
    return set(' '.join(w[i:i+3]) for i in range(max(1,len(w)-2)))
def deduplicate(df):
    df=df.copy(); df['normalized']=df.text_.map(normalize)
    df=df[df.normalized.str.len()>0]
    conflicts=df.groupby('normalized').label.nunique()
    df=df[~df.normalized.isin(conflicts[conflicts>1].index)].drop_duplicates('normalized').reset_index(drop=True)
    parent=list(range(len(df)))
    def find(a):
        while parent[a]!=a: parent[a]=parent[parent[a]]; a=parent[a]
        return a
    lsh=MinHashLSH(threshold=.7,num_perm=128)
    sets=[]; matches=0
    for i,s in enumerate(df.normalized):
        tokens=shingles(s); sets.append(tokens); h=MinHash(num_perm=128,seed=SEED)
        h.update_batch([t.encode() for t in sorted(tokens)])
        for k in lsh.query(h):
            j=int(k)
            if len(tokens & sets[j])/len(tokens | sets[j]) >= .85:
                parent[find(i)]=find(j); matches+=1
        lsh.insert(str(i),h)
        if i%10000==0: print('Deduplication',i,flush=True)
    df['group']=[find(i) for i in range(len(df))]
    conflicts=df.groupby('group').label.nunique()
    df=df[~df.group.isin(conflicts[conflicts>1].index)].drop_duplicates('group').reset_index(drop=True)
    return df,matches

def metrics(y,p):
    return dict(accuracy=accuracy_score(y,p),precision=precision_score(y,p,zero_division=0),recall=recall_score(y,p,zero_division=0),f1=f1_score(y,p,zero_division=0),confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist(),per_class=classification_report(y,p,labels=[0,1],target_names=['OR','CG'],output_dict=True,zero_division=0))
def main():
    raw=pd.read_csv(ROOT/'data/reviews.csv')
    assert set(raw.label)=={'OR','CG'}
    df,pairs=deduplicate(raw)
    y=(df.label=='CG').astype(int)
    ids=np.arange(len(df))
    train,rest=train_test_split(ids,test_size=.4,random_state=SEED,stratify=y)
    val,test=train_test_split(rest,test_size=.5,random_state=SEED,stratify=y.iloc[rest])
    models={
      'majority_baseline':DummyClassifier(strategy='most_frequent'),
      'logistic_regression':LogisticRegression(C=1,max_iter=1500,random_state=SEED),
      'linear_svm':LinearSVC(C=1,random_state=SEED,max_iter=3000)}
    fitted={}; validation={}
    for name,clf in models.items():
        pipe=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,sublinear_tf=True,max_features=60000)),('classifier',clf)])
        pipe.fit(df.text_.iloc[train],y.iloc[train]); fitted[name]=pipe
        validation[name]=metrics(y.iloc[val],pipe.predict(df.text_.iloc[val]))
        print('Validation',name,validation[name]['f1'],flush=True)
    chosen=max(['logistic_regression','linear_svm'],key=lambda n:validation[n]['f1'])
    # Keep the exact fitted model used for evaluation: no silent refit on test data.
    results={n:metrics(y.iloc[test],p.predict(df.text_.iloc[test])) for n,p in fitted.items()}
    categories={c:metrics(y.iloc[test][df.category.iloc[test]==c],fitted[chosen].predict(df.text_.iloc[test][df.category.iloc[test]==c])) for c in sorted(df.category.unique())}
    report=dict(dataset_rows=len(raw),retained_rows=len(df),removed_rows=len(raw)-len(df),near_duplicate_pairs=pairs,split_sizes=dict(train=len(train),validation=len(val),test=len(test)),seed=SEED,positive_label='CG',selected_model=chosen,selection='Highest validation CG F1; fixed C=1; threshold=0; no test tuning',validation=validation,test=results,category_test=categories,score='Uncalibrated decision-function margin; positive values favor CG. Not probability or confidence.',dataset_sha256=hashlib.sha256((ROOT/'data/reviews.csv').read_bytes()).hexdigest(),versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,pandas=pd.__version__),deduplication='Normalized exact deduplication; MinHash LSH candidate search (128 permutations, threshold .70), verified 3-word-shingle Jaccard >= .85; connected components collapsed; conflicting-label components removed. Approximate search can miss near-duplicates. Product/user IDs unavailable; product-disjoint generalization not established.')
    (ROOT/'models').mkdir(exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump(fitted[chosen],ROOT/'models/model.joblib',compress=3)
    (ROOT/'reports/metrics.json').write_text(json.dumps(report,indent=2))
    split_map={int(i):s for s,a in [('train',train),('validation',val),('test',test)] for i in a}
    pd.DataFrame({'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in df.normalized],'label':df.label,'split':[split_map[i] for i in ids]}).to_csv(ROOT/'reports/split_manifest.csv',index=False)
    print(json.dumps({n:{k:v for k,v in m.items() if k in ['accuracy','precision','recall','f1','confusion_matrix']} for n,m in results.items()},indent=2),flush=True)
if __name__=='__main__': main()
