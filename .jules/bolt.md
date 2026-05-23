## 2025-02-28 - Optimize CVXPY canonicalization
**Learning:** Replacing element-wise multiplications with vector inner products drastically reduces canonicalization time without degrading solver execution time.
**Action:** Use inner products for weighted norms instead of cp.sum(cp.multiply(...)).
