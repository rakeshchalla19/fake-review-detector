import io
import numpy as np
import pytest
import app as service

TEXT='I used these headphones for two weeks. The sound is clear but the cable feels thin.'
@pytest.fixture
def client():
    service.app.config['TESTING']=True
    service.requests_window.clear()
    return service.app.test_client()

def test_inference_matches_pipeline(client):
    r=client.post('/api/analyze',json={'review':TEXT,'product':'Headphones'})
    assert r.status_code==200
    a=r.json['results'][0]; margin=float(service.model.decision_function([TEXT])[0])
    assert a['score']==round(margin,4)
    assert a['flagged']==bool(margin>=0)
    assert a['label']==('Potentially Fake' if margin>=0 else 'Likely Genuine')
    assert a['top_terms']

def test_explanation_reconstructs_margin():
    x=service.model.named_steps['tfidf'].transform([TEXT])
    clf=service.model.named_steps['classifier']
    assert np.isclose((x@clf.coef_[0])[0]+clf.intercept_[0],service.model.decision_function([TEXT])[0])

@pytest.mark.parametrize('payload',[{},[],{'review':''},{'review':23},{'review':'short'},{'review':'x'*5001},{'review':TEXT,'product':42},{'review':TEXT,'product':'x'*121},{'review':'Este producto es muy bueno y funciona perfectamente para mis necesidades diarias.'},{'review':'यह उत्पाद बहुत अच्छा है और मुझे यह बहुत पसंद आया है।'}])
def test_invalid(client,payload): assert client.post('/api/analyze',json=payload).status_code==400

def upload(client,content,name='data.csv'):
    return client.post('/api/bulk',data={'file':(io.BytesIO(content),name)},content_type='multipart/form-data')

def test_bulk_sample_summary_similarity(client):
    data=client.get('/sample.csv').data
    r=upload(client,data)
    assert r.status_code==200 and r.json['summary']['total']==3
    assert len(r.json['summary']['products'])==2
    assert r.json['results'][0]['nearest_row']==2
    assert r.json['results'][0]['similarity']>.8

@pytest.mark.parametrize('content,name',[(b'wrong\nx','data.csv'),(b'review\n','data.csv'),(b'\xff','data.csv'),(b'review\nx','data.txt'),(b'review,review\na,b','data.csv'),(b'review\n"unclosed','data.csv'),(b'review,product\nhello','data.csv'),(b'review,other\nx,y','data.csv')])
def test_bad_csv(client,content,name): assert upload(client,content,name).status_code==400

def test_row_and_size_limits(client):
    assert upload(client,('review\n'+(TEXT+'\n')*201).encode()).status_code==400
    assert upload(client,b'a'*(1024*1024+1)).status_code==413

def test_health_security_and_no_store(client):
    assert client.get('/health').json['status']=='ok'
    for path in ['/','/analyze','/bulk','/dashboard','/about']:
        r=client.get(path); assert r.status_code==200
        assert r.headers['Cache-Control']=='no-store'
        assert "script-src 'self'" in r.headers['Content-Security-Policy']

def test_rate_limit(client,monkeypatch):
    monkeypatch.setattr(service,'LIMIT',1)
    client.post('/api/analyze',json={'review':TEXT})
    assert client.post('/api/analyze',json={'review':TEXT}).status_code==429

def test_model_missing(client,monkeypatch):
    monkeypatch.setattr(service,'model',None)
    assert client.get('/health').status_code==503
    assert client.post('/api/analyze',json={'review':TEXT}).status_code==503

def test_user_html_is_not_rendered_as_html(client):
    product='<img src=x onerror=alert(1)>'
    r=client.post('/api/analyze',json={'review':TEXT,'product':product})
    assert r.json['results'][0]['product']==product
    # Client uses textContent, never HTML insertion for review/product content.
    js=(service.ROOT/'static/app.js').read_text()
    assert 'innerHTML' not in js and 'textContent' in js
