## 2024-05-24 - Avoid explicit intermediate nodes in CVXPY aggregations
**Learning:** CVXPY compilation time can degrade significantly due to explicit intermediate graph nodes created by `cp.multiply` followed by aggregations like `cp.sum`, `cp.sum_squares`, or `cp.norm1`.
**Action:** Replace `cp.sum(cp.multiply(...))` and similar constructs with direct inner products (`@`) applying appropriate mathematical equivalents like `np.square(weights) @ cp.sum(cp.square(beta_var), axis=1)` to dramatically reduce expression tree complexity.
