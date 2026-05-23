## 2025-05-23 - Optimize CVXPY canonicalization
**Learning:** Replacing element-wise multiplication and summation (e.g. cp.sum(cp.multiply(A, B))) with a vector inner product (A @ B) drastically reduces CVXPY compilation time without degrading solver time.
**Action:** Use inner products instead of cp.multiply for vector operations in objective penalization terms.
