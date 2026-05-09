## 2024-05-09 - Modernizer Initialized
**Learning:** Understanding CVXPY constraints and optimizations.
**Action:** Replace `cp.sum(cp.multiply(weights, norms))` with `weights.T @ norms` or `weights @ norms` for faster canonicalization without losing readability.
