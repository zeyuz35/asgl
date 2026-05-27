## 2025-05-27 - Optimize CVXPY canonicalization
**Learning:** Element-wise multiplications followed by summation or norms (e.g. `cp.norm1(cp.multiply(W, B))`) create large canonicalization trees.
**Action:** Replace them with vector inner products (e.g. `np.abs(W).reshape(-1) @ cp.abs(B)`) to heavily shrink compiled trees and reduce canonicalization time.
