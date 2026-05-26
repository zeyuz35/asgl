## 2024-05-13 - Optimize CVXPY Expressions
**Learning:** For CVXPY performance optimization, replacing element-wise multiplications followed by summation (`cp.sum(cp.multiply(A, B))`) with a vector inner product (`A @ B`) drastically reduces canonicalization time without degrading solver execution time by heavily shrinking the compiled expression tree.
**Action:** Replace `cp.sum(cp.multiply(..., ...))` with `... @ ...` and `cp.norm1(cp.multiply(A, B))` with `A.T @ cp.abs(B)`.
