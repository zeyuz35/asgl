# Modernizer's Journal

## 2024-05-18 - Optimize CVXPY canonicalization
**Learning:** CVXPY canonicalization is extremely slow when chaining `cp.multiply` and `cp.sum`, but can be optimized by using matrix multiplication (`@`). Note that `cp.sum_squares(cp.multiply(...))` is already highly optimized and should NOT be replaced. Replacing `cp.norm1(cp.multiply(W, B))` with `cp.sum(W.T @ cp.abs(B))` and `cp.sum(cp.multiply(W, B))` with `W.T @ B` vastly reduces compilation time while remaining DCP compliant.
**Action:** When working with CVXPY problems, proactively avoid element-wise multiplications (`cp.multiply`) paired with aggregations, preferring vectorized inner products (`@`) to improve formulation performance, but preserve `cp.sum_squares(cp.multiply(...))`.
