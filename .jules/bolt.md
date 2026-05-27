## 2024-05-27 - Optimize CVXPY canonicalization time
**Learning:** Replacing element-wise multiplications followed by summation (`cp.sum(cp.multiply(A, B))`) or norms (`cp.norm1(cp.multiply(A, B))`) with vector inner products (`A @ B` or `np.abs(A).reshape(-1) @ cp.abs(B)`) drastically reduces canonicalization time without degrading solver execution time by heavily shrinking the compiled expression tree.
**Action:** Always prefer vector inner products over element-wise multiplication and summation/norms when building CVXPY expressions.
