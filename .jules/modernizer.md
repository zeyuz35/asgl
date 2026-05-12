## 2025-05-12 - Optimize CVXPY canonicalization
**Learning:** CVXPY canonicalizes `cp.sum(cp.multiply(weights, beta_var))` much slower than an equivalent dot product `weights @ beta_var` or `cp.sum(weights.T @ cp.abs(beta_var))`. Replacing element-wise multiplication with inner products drastically speeds up expression construction.
**Action:** Always replace `cp.sum(cp.multiply(weights, norms))` with `weights @ norms` when `weights` are non-negative, and map other `cp.multiply` aggregations (`sum_squares`, `norm1`) to their matrix multiplication equivalents.
