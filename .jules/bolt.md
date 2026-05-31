## 2023-10-25 - CVXPY Canonicalization Optimization
**Learning:** CVXPY canonicalization time for weighted norms and sums of multiplied variables can be significantly reduced by replacing element-wise cp.multiply followed by a summation or norm with a direct vector inner product (e.g. replacing cp.norm1(cp.multiply(W, B)) with cp.sum(np.abs(W).T @ cp.abs(B))). This shrinks the expression tree dramatically.
**Action:** Always prefer vector inner products over element-wise multiplication in large CVXPY objective functions, while strictly wrapping constants in np.abs() to satisfy DCP rules.
