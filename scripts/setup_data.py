"""Download the FlyWire connectome files (pinned + checksummed) and build the 3D viewer assets.

    python scripts/setup_data.py            # ~135 MB download, one time

Sources:
  * philshiu/Drosophila_brain_model (MIT): FlyWire v783 connectivity in the format of Shiu et al. 2024
  * flyconnectome/flywire_annotations: neuron annotations (cell types, positions); please cite (see README)
  * FlyWire public neuropil mesh (Google Cloud Storage) for the brain outline in the viewer
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import requests

from flytrader import paths

ROOT = Path(__file__).resolve().parent.parent
DEST = paths.DATA / "connectome"

SHIU = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNO = "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/a83b2776d60d5764cef36b927f5f9679c16c47a2"

FILES = {
    "Completeness_783.csv": (f"{SHIU}/Completeness_783.csv",
                             "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311"),
    "Connectivity_783.parquet": (f"{SHIU}/Connectivity_783.parquet",
                                 "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"),
    "annotations.tsv": (f"{ANNO}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
                        "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, digest: str):
    path = DEST / name
    if path.exists() and sha256(path) == digest:
        print(f"  ok      {name}", flush=True)
        return
    print(f"  get     {name}", flush=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    got = sha256(tmp)
    if got != digest:
        tmp.unlink()
        raise SystemExit(f"checksum mismatch for {name}: expected {digest}, got {got}")
    tmp.rename(path)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print("connectome data:", flush=True)
    for name, (url, digest) in FILES.items():
        fetch(name, url, digest)
    print("viewer assets:", flush=True)
    subprocess.run([sys.executable, str(ROOT / "viz" / "export_assets.py")], check=True, cwd=ROOT)
    print("done")


if __name__ == "__main__":
    main()
