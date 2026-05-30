## 2025-05-30 - Modernize CVXPY expressions with inner products
**Learning:** In CVXPY, `cp.sum(cp.multiply(A, B))` and `cp.norm1(cp.multiply(A, B))` canonicalize much slower than their inner product counterparts `A @ B` and `np.abs(A).reshape(-1) @ cp.abs(B)`.
**Action:** Replaced `cp.multiply` followed by norms/sums with dot products to optimize canonicalization time while preserving readability.
