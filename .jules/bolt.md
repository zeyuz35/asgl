## 2024-06-10 - Optimize inner products in CVXPY canonicalization
**Learning:** CVXPY's `cp.sum(cp.multiply(x, y))` or equivalent combinations are slower during canonicalization and solver execution than using an inner product directly, like `x @ y` or `x.T @ y` for arrays/variables.
**Action:** Replaced element-wise multiplication followed by summing with matrix multiplication/inner product using the `@` operator across the logistic objective and group penalty definitions.
