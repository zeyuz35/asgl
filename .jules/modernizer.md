## 2024-06-05 - Modernizer init
**Learning:** CVXPY optimization expressions use deprecated cp.multiply for inner products, which can be optimized.
**Action:** Replace `cp.sum(cp.multiply(A, B))` with inner products `A @ B`.
