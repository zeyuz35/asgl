## 2024-05-19 - Modernize CVXPY Element-Wise Multiplication
**Learning:** CVXPY canonicalization can be drastically sped up by replacing `cp.sum(cp.multiply(A, B))` with inner product `A @ B` without negatively impacting actual solver time.
**Action:** Use vector inner products for sums of element-wise multiplications, especially for logistic loss and grouped penalty norms.
