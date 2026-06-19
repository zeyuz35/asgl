## YYYY-MM-DD - Optimize CVXPY inner products
**Learning:** CVXPY elementwise-multiplication followed by sum (`cp.sum(cp.multiply(a, b))`) is significantly slower than using the matrix multiplication operator (`a @ b`) for 1D arrays.
**Action:** Use `@` for inner products of vectors in CVXPY to improve performance without changing mathematical semantics or DCP compliance.
