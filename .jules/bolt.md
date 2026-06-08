## YYYY-MM-DD - Optimize CVXPY canonicalization
**Learning:** CVXPY canonicalization is significantly faster when using inner products instead of element-wise multiplication followed by summing or sum_squares, especially for large matrices.
**Action:** Replace cp.sum_squares(cp.multiply(A, B)) and cp.sum(cp.multiply(A, B)) with inner products where possible to reduce expression tree sizes.
