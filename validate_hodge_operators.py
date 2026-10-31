from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.linalg import eigsh

OUT = Path("processed")

B1 = load_npz(OUT / "TENX53_B1.npz").tocsr()
B2 = load_npz(OUT / "TENX53_B2.npz").tocsr()
L1 = load_npz(OUT / "TENX53_L1.npz").tocsr()

print("HODGE OPERATOR VALIDATION")
print("=" * 40)

# 1. Matrix dimensions
print(f"B1 shape: {B1.shape}")
print(f"B2 shape: {B2.shape}")
print(f"L1 shape: {L1.shape}")

# 2. Boundary identity: B1 @ B2 = 0
boundary = B1 @ B2
boundary.eliminate_zeros()

print(f"\nB1 @ B2 has {boundary.nnz} nonzero entries")
assert boundary.nnz == 0, "Boundary identity failed"

# 3. Symmetry of the Hodge 1-Laplacian
asymmetry = L1 - L1.T
asymmetry.eliminate_zeros()

print(f"L1 symmetry difference nnz: {asymmetry.nnz}")
assert asymmetry.nnz == 0, "L1 is not symmetric"

# 4. Check smallest algebraic eigenvalues
# L1 is positive semidefinite in exact arithmetic.
eigenvalues = eigsh(
    L1.astype(np.float64),
    k=6,
    which="SA",
    return_eigenvectors=False,
    tol=1e-7,
)

eigenvalues = np.sort(eigenvalues)

print("\nSix smallest algebraic eigenvalues:")
for i, value in enumerate(eigenvalues, start=1):
    print(f"{i}: {value:.10e}")

tol = 1e-7
if eigenvalues[0] < -tol:
    raise ValueError(
        "A materially negative eigenvalue was detected."
    )

print("\nStructural checks passed.")
print("Note: six eigenvalues do not establish the full nullity.")