## 2024-05-20 - Optimize CVXPY canonicalization
**Learning:** CVXPY canonicalizes `cp.sum(cp.multiply(A, B))` and `cp.norm1(cp.multiply(weights, beta_var))` poorly, leading to significantly large problem sizes. Using vector inner products like `A @ B` or `weights.T @ cp.abs(beta_var)` drastically shrinks the expression tree, speeding up problem canonicalization and maintaining or improving solver execution time.
**Action:** When working with element-wise multiplication inside sums or L1-norms in CVXPY, prefer using inner products and `@` where mathematically equivalent.
