## 2024-05-24 - Optimize CVXPY canonicalization
**Learning:** CVXPY's `cp.multiply` followed by sum/norm can drastically slow down canonicalization. Vector inner products are functionally identical but much faster to canonicalize.
**Action:** Replace element-wise multiply-then-sum patterns (`cp.sum(cp.multiply(a, b))` -> `a @ b`, `cp.norm1(cp.multiply(a, b))` -> `cp.sum(a.T @ cp.abs(b))`) for CVXPY optimization without altering math.
