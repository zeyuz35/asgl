## 2024-05-25 - CVXPY Performance Modernization
**Learning:** For CVXPY performance optimization, replacing element-wise multiplications followed by summation (`cp.sum(cp.multiply(A, B))`) with a vector inner product (`A @ B`) drastically reduces canonicalization time without degrading solver execution time by heavily shrinking the compiled expression tree.
**Action:** Use native `@` operator for inner products instead of `cp.sum(cp.multiply(...))` where applicable.
