"""Download the public All_Beauty review corpus from the Amazon Reviews '23 project."""
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz"
DEST = ROOT / "data" / "amazon-reviews-2023-All_Beauty.jsonl.gz"

def main():
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"Dataset already exists: {DEST}")
        return
    request = urllib.request.Request(URL, headers={"User-Agent": "ReviewLens research project"})
    with urllib.request.urlopen(request, timeout=180) as response, DEST.open("wb") as output:
        while chunk := response.read(1024 * 1024): output.write(chunk)
    print(f"Downloaded dataset: {DEST} ({DEST.stat().st_size:,} bytes)")

if __name__ == "__main__": main()
