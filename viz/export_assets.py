"""Export static 3D assets for the brain viewer: neuron positions, classes, brain shell."""

import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from flytrader import paths

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viz" / "assets"
CON = paths.DATA / "connectome"
SHELL_URL = "https://storage.googleapis.com/flywire_neuropil_meshes/whole_neuropil/brain_mesh_v141.surf/mesh/1:0:0"
CLASSES = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
           "sensory_ascending", "visual_centrifugal", "motor", "endocrine"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(CON / "Completeness_783.csv", index_col=0).index.values.astype(np.int64)
    a = pd.read_csv(CON / "annotations.tsv", sep="\t", low_memory=False).drop_duplicates("root_id").set_index("root_id")
    a = a.reindex(ids)
    vox = np.where(a[["soma_x"]].notna().values, a[["soma_x", "soma_y", "soma_z"]].values,
                   a[["pos_x", "pos_y", "pos_z"]].values).astype(np.float64)
    nm = vox * np.array([4, 4, 40])

    r = requests.get(SHELL_URL, timeout=60)
    r.raise_for_status()
    buf = r.content
    nv = struct.unpack("<I", buf[:4])[0]
    verts = np.frombuffer(buf[4:4 + nv * 12], dtype="<f4").reshape(-1, 3).astype(np.float64)
    faces = np.frombuffer(buf[4 + nv * 12:], dtype="<u4")
    center = (verts.min(0) + verts.max(0)) / 2

    def to_scene(p):
        q = (p - center) / 500_000.0  # nm -> scene units, brain spans about +/-0.7
        q[:, 1] *= -1  # FlyWire y points ventral
        q[:, 2] *= -1
        return q.astype(np.float32)

    pts = to_scene(np.nan_to_num(nm, nan=center[0]))
    cls = a.super_class.map({c: i for i, c in enumerate(CLASSES)}).fillna(255).astype(np.uint8).values
    pts.tofile(OUT / "neurons.bin")
    cls.tofile(OUT / "class.bin")
    sv = to_scene(verts)
    np.concatenate([np.array([len(sv), len(faces)], dtype=np.uint32).view(np.float32),
                    sv.ravel()]).astype(np.float32).tofile(OUT / "shell_verts.bin")
    faces.astype(np.uint32).tofile(OUT / "shell_faces.bin")

    from flytrader.circuits import BITTER, MN9, SUGAR
    idx = {int(r): i for i, r in enumerate(ids)}
    ct = a.cell_type.fillna("")
    groups = {
        "sugar": [idx[r] for r in SUGAR if r in idx],
        "bitter": [idx[r] for r in BITTER if r in idx],
        "mn9": [idx[r] for r in MN9 if r in idx],
        "giant_fiber": list(np.flatnonzero(ct.values == "DNp01")),
        "looming": list(np.flatnonzero(ct.isin(["LC4", "LPLC2"]).values)),
        "mdn": list(np.flatnonzero(ct.values == "MDN")),
        "kenyon": list(np.flatnonzero((a.cell_class == "Kenyon_Cell").values)),
        "mbon": list(np.flatnonzero((a.cell_class == "MBON").values)),
        "dan": list(np.flatnonzero((a.cell_class == "DAN").values)),
    }
    (OUT / "labels.json").write_text(json.dumps({"classes": CLASSES, "n": len(ids),
        "groups": {k: [int(i) for i in v] for k, v in groups.items()}}))
    print("neurons", len(ids), "shell verts", len(sv), "faces", len(faces) // 3,
          {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
