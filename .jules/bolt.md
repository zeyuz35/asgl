## YYYY-MM-DD - Optimize CVXPY canonicalization
**Learning:** CVXPY canonicalization for element-wise multiplication followed by summation is significantly slower than using matrix multiplication due to parsing overhead.
**Action:** Replace aggregations of element-wise array operations with vectorized matrix inner products to improve compilation and canonicalization speed.
