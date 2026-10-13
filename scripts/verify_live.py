"""Run only against a deployment you own. Verifies health, real inference and CSV route."""
import sys, urllib.request, json, uuid
base=sys.argv[1].rstrip('/')
if not base.startswith('https://'): raise SystemExit('Use the actual HTTPS deployment URL.')
def call(path,data=None,content_type=None):
    headers={'Content-Type':content_type} if content_type else {}
    with urllib.request.urlopen(urllib.request.Request(base+path,data=data,headers=headers),timeout=120) as r:
        return r.read()
assert json.loads(call('/health'))['status']=='ok'
r=json.loads(call('/api/analyze',json.dumps({'review':'I used these headphones for two weeks. The sound is clear but the cable feels thin.','product':'Headphones'}).encode(),'application/json'))
assert r['summary']['total']==1 and isinstance(r['results'][0]['score'],(int,float))
sample=call('/sample.csv'); boundary=uuid.uuid4().hex
body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="sample.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode()+sample+f'\r\n--{boundary}--\r\n'.encode())
r=json.loads(call('/api/bulk',body,'multipart/form-data; boundary='+boundary))
assert r['summary']['total']==3 and len(r['summary']['products'])==2
print('Verified HTTPS health, single inference, and sample CSV:',base)
