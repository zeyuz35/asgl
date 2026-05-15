## 2024-05-16 - CVXPY Performance optimization
**Learning:** `cp.sum(cp.multiply(A, B))` and `cp.norm1(cp.multiply(W, B))` take O(N) constraints to canonicalize in CVXPY. Replacing them with inner products `A @ B` and `W.T @ cp.abs(B)` reduces the AST size drastically and yields a huge performance boost for large inputs (30-50%).
**Action:** Use `A @ B` instead of `cp.sum(cp.multiply(A, B))` in `_gl`, `_sgl`, `_agl`, `_asgl`, and `_define_objective_function`. Use `W.T @ cp.abs(B)` instead of `cp.norm1(cp.multiply(W, B))` for adaptive individual weights.
