## 2024-05-14 - CVXPY Canonicalization Optimization
**Learning:** CVXPY canonicalization time can be drastically reduced by replacing element-wise multiplications followed by summation (`cp.sum(cp.multiply(A, B))`) with a vector inner product (`A @ B`). This heavily shrinks the compiled expression tree without degrading solver execution time.
**Action:** When working with CVXPY problems, especially those involving large penalization terms or weights, always prefer matrix multiplications or dot products over element-wise multiplications combined with summations.
