## 2025-02-12 - CVXPY Expression Optimization
**Learning:** CVXPY canonicalization scales poorly with element-wise multiplication followed by summation (`cp.sum(cp.multiply(A, B))`), but handles inner products (`A @ B`) natively and much more efficiently.
**Action:** Replace `cp.sum(cp.multiply(A, B))` with `A @ B` wherever possible, taking care to transpose the first argument if it's a 2D matrix (`A.T @ cp.abs(B)` instead of `cp.norm1(cp.multiply(A, B))`). Avoid converting `cp.sum_squares(cp.multiply(A, B))` as `cp.sum_squares` itself is already highly optimized.
