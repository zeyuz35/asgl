## YYYY-MM-DD - [Optimize CVXPY inner products]
**Learning:** Replacing element-wise multiplication followed by sum with a direct inner product significantly reduces canonicalization time in CVXPY graph construction, especially for grouped features or large matrices.
**Action:** Use vector/matrix multiplications instead of cp.multiply when aggregating penalized terms or calculating logit errors.
