## 2025-02-14 - Replace cvxpy multiply with inner products for canonicalization speed
**Learning:** cvxpy's `cp.multiply` canonicalization scales poorly. Inner products using `@` drastically reduce canonicalization times without affecting solver time. Constants must be explicitly strictly positive using `np.abs()` when multiplying convex functions like `cp.abs()`.
**Action:** Always prefer `@` over `cp.sum(cp.multiply(...))` or `cp.norm1(cp.multiply(...))` for weighted sums.
