## 2024-06-11 - Optimize Logistic Regression Objective Canonicalization
**Learning:** CVXPY canonicalization for logistic regression can be slow when using `cp.sum(cp.logistic(pred) - cp.multiply(y, pred))`. Replacing `cp.multiply` with an inner product (matrix multiplication via `@`) significantly speeds up canonicalization, especially for flat arrays.
**Action:** Use inner products instead of element-wise multiplication with a subsequent sum when aggregating arrays in CVXPY to improve performance.
