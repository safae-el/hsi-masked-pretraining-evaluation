"""Download the classic benchmark scenes into --root (default ~/hsi_data)."""
import argparse
import urllib.request
from pathlib import Path

BASE = "https://www.ehu.eus/ccwintco/uploads"
FILES = {
    "Indian_pines_corrected.mat": f"{BASE}/6/67/Indian_pines_corrected.mat",
    "Indian_pines_gt.mat":        f"{BASE}/c/c4/Indian_pines_gt.mat",
    "PaviaU.mat":                 f"{BASE}/e/ee/PaviaU.mat",
    "PaviaU_gt.mat":              f"{BASE}/5/50/PaviaU_gt.mat",
    "Salinas_corrected.mat":      f"{BASE}/a/a3/Salinas_corrected.mat",
    "Salinas_gt.mat":             f"{BASE}/f/fa/Salinas_gt.mat",
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/hsi_data")
    root = Path(ap.parse_args().root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dst = root / name
        if dst.exists():
            print("exists:", name); continue
        print("downloading:", name)
        urllib.request.urlretrieve(url, dst)
    print("done ->", root)
