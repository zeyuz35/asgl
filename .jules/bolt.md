## 2024-05-17 - Optimize CVXPY canonicalization
**Learning:** For CVXPY performance optimization, replacing element-wise multiplications followed by summation (`cp.sum(cp.multiply(A, B))`) or norms (`cp.norm1(cp.multiply(A, B))`) with a vector inner product (`A @ B` or `np.abs(A).reshape(-1) @ cp.abs(B)`) drastically reduces canonicalization time without degrading solver execution time by heavily shrinking the compiled expression tree.
**Action:** Replace these operations with vector inner products when constructing penalization objectives in cvxpy.
