"""Download the original OSF release and verify its publisher-supplied SHA-256."""
from pathlib import Path
import hashlib, urllib.request
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://osf.io/download/3vds7/'
SHA256 = 'ea03422988824bc917b5f7217e3867e27b597c647d5261adfae73cd7a85e1e1a'
def main():
    path = ROOT / 'data/reviews.csv'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=120) as r:
            content = r.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError('Dataset checksum mismatch; inspect the upstream release before proceeding.')
        path.write_bytes(content)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256, 'Dataset checksum mismatch'
    print('Verified dataset:', path)
if __name__ == '__main__': main()
