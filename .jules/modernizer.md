## 2024-05-31 - Optimize CVXPY canonicalization
**Learning:** For CVXPY, replacing `cp.norm1(cp.multiply(A, B))` and `cp.sum(cp.multiply(A, B))` with inner products `np.abs(A) @ cp.abs(B)` and `A @ B` improves canonicalization time without degrading solver execution.
**Action:** Replace element-wise multiplications followed by summation or norms with vector inner products.
