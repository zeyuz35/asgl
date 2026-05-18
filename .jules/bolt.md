## 2024-05-18 - Optimize CVXPY Expressions
**Learning:** Element-wise multiplications followed by summation or norm operations in CVXPY (e.g., `cp.sum(cp.multiply(A, B))`, `cp.norm1(cp.multiply(W, B))`) drastically increase canonicalization time due to building complex expression trees.
**Action:** Replace these patterns with vector inner products (e.g., `A @ B`, `cp.sum(W.T @ cp.abs(B))`). This shrinks canonicalization time and tree size while either maintaining or improving solve time. (But avoid replacing `cp.sum_squares(cp.multiply(W, B))` as `cp.sum_squares` handles it efficiently).
