"""Download PaySim (6.36M synthetic mobile-money transactions, ~470 MB) from Zenodo.

Source record: https://zenodo.org/records/22761688 (CC-BY-4.0), a mirror of the
Kaggle "ealaxi/paysim1" release by E. Lopez-Rojas.
"""
import sys
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/api/records/22761688/files/PS_20174392719_1491204439457_log.csv/content"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "paysim.csv"
EXPECTED_ROWS = 6_362_620


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, DEST)
    with DEST.open("rb") as f:
        rows = sum(1 for _ in f) - 1
    print(f"saved {DEST} ({rows:,} rows)")
    return 0 if rows == EXPECTED_ROWS else 1


if __name__ == "__main__":
    sys.exit(main())
