## 2024-05-13 - Optimize cvxpy expression compilation
**Learning:** In cvxpy, replacing element-wise multiplication followed by sum or norm1 with vector inner products drastically reduces canonicalization time without degrading solver execution time.
**Action:** Replaced cp.norm1(cp.multiply(weights, beta_var)) and cp.sum(cp.multiply(group_weights, group_norms)) with matrix multiplication equivalents.
