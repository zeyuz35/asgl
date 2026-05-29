## 2024-05-14 - Vectorize CVXPY operations
**Learning:** Replacing element-wise multiplication and summation (`cp.sum(cp.multiply(W, B))`) or norm (`cp.norm1(cp.multiply(W, B))`) with a vector inner product (`W @ B` or `W.T @ cp.abs(B)`) drastically reduces canonicalization time without degrading solver execution time.
**Action:** Always prefer vector inner products over element-wise multiplication with summation/norm for CVXPY optimization unless explicit element-wise constraints are required.
