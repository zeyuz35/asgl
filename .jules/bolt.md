## 2024-05-28 - Optimize CVXPY canonicalization using vector inner products
**Learning:** Replacing element-wise multiplications followed by summation or norms (`cp.norm1(cp.multiply(A, B))` or `cp.sum(cp.multiply(A, B))`) with vector inner products drastically reduces canonicalization time without degrading solver execution time.
**Action:** Use vector inner products for weighted norms in CVXPY expression trees to avoid complex element-wise constraints.
