## 2024-05-24 - Optimize CVXPY canonicalization
**Learning:** Element-wise multiplication followed by sum or norm operations (`cp.sum(cp.multiply(A, B))`) drastically slows down CVXPY's canonicalization process by inflating the expression tree.
**Action:** Replace `cp.sum(cp.multiply(A, B))` with inner products (e.g. `A @ B`). For operations involving constants and variables, ensure DCP compliance by wrapping the constant weights in `np.abs()` if required for norms or squaring them for `sum_squares`.
