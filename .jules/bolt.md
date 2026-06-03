## 2024-06-03 - Optimize logistic regression objective canonicalization
**Learning:** CVXPY canonicalizes `cp.sum(cp.logistic(x) - cp.multiply(y, x))` slower than `cp.sum(cp.logistic(x)) - y.T @ x`. The element-wise multiplication constraint introduces an unnecessary intermediate representation that slows down the parser and the subsequent solver.
**Action:** When finding `cp.sum(cp.logistic(x) - cp.multiply(y, x))`, explicitly check if variables are 1D arrays and replace the element-wise multiplication with a vector inner product `y.T @ x`.
