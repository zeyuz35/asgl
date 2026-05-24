## 2025-03-01 - Modernize CVXPY sum(multiply(...)) to inner product
**Learning:** CVXPY element-wise multiplication followed by summation (`cp.sum(cp.multiply(A, B))`) is less efficient than vector inner products (`A.T @ B` or `A @ B`) which are faster and reduce canonicalization time.
**Action:** Replace `cp.sum(cp.multiply(A, B))` with `A.T @ B` or `A @ B` (for 1D arrays, `A @ B` is sufficient) to modernize performance.
