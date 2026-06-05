## 2024-06-05 - Optimize CVXPY canonicalization using inner products
**Learning:** CVXPY canonicalization time can be significantly improved by replacing sum of multiplied terms (or norm1 of multiplied terms) with inner products, shrinking the canonicalization graph without affecting solver execution times. For logistic objectives, using the inner product is safe when flattening to 1D arrays beforehand.
**Action:** Replace `cp.sum(cp.multiply(A, B))` and `cp.norm1(cp.multiply(weights, vars))` with vector inner products (`A @ B` and `np.abs(weights) @ cp.abs(vars)`) where applicable.
