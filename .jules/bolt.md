## 2024-05-25 - Optimize CVXPY canonicalization
**Learning:** Using `cp.sum(cp.multiply(A, B))` for 1D arrays introduces element-wise multiplication overhead during canonicalization, whereas the vector inner product `A.T @ B` dramatically shrinks the compiled expression tree and reduces canonicalization time without impacting solver execution time.
**Action:** Replace `cp.sum(cp.multiply(A, B))` with `A.T @ B` for vector dot products in CVXPY objectives.
