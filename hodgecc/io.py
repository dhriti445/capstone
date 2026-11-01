"""Save and load a cell complex so teammates can reuse it.

File format (.npz, no pickle):
  edges      (E, 2) int64, i < j, sorted
  triangles  (T, 3) int64, a < b < c, sorted
  spot_ids   (N,) str, spot barcodes in the row order the indices refer to
  sample     str
Node index k in edges/triangles = row k of the .h5ad the complex was built
from (after keeping only in_tissue == 1 spots).
"""
import numpy as np
from hodgecc.complex import build_incidence, check_boundary


def save_complex(path, edges, triangles, spot_ids, sample=""):
    np.savez_compressed(
        path,
        edges=np.asarray(edges, dtype=np.int64),
        triangles=np.asarray(triangles, dtype=np.int64),
        spot_ids=np.asarray(spot_ids).astype(str),
        sample=np.asarray(str(sample)),
    )


def load_complex(path, adata=None):
    """Return (edges, triangles, B1, B2). If adata is given, check that its
    spots are the same spots in the same order."""
    z = np.load(path, allow_pickle=False)
    edges, triangles = z["edges"], z["triangles"]
    spot_ids = z["spot_ids"]
    if adata is not None:
        if "in_tissue" in adata.obs:
            adata = adata[adata.obs["in_tissue"] == 1]
        ids = np.asarray(adata.obs_names).astype(str)
        if len(ids) != len(spot_ids) or not (ids == spot_ids).all():
            raise ValueError("adata spots do not match the complex file; "
                             "use the same .h5ad the complex was built from")
    B1, B2 = build_incidence(edges, triangles, len(spot_ids))
    assert check_boundary(B1, B2), "B1 @ B2 != 0"
    return edges, triangles, B1, B2
