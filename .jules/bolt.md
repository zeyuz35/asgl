## YYYY-MM-DD - Optimize CVXPY canonicalization with inner products
**Learning:** Replacing element-wise multiplications followed by summation or norms (`cp.sum(cp.multiply(A, B))` or `cp.norm1(cp.multiply(A, B))`) with a vector inner product (`A @ B` or `np.abs(A).reshape(-1) @ cp.abs(B)`) drastically reduces CVXPY canonicalization time.
**Action:** Always prefer vector inner products over element-wise multiplication with summation for performance in CVXPY, while ensuring proper dimensionality and absolute values to satisfy DCP rules.
