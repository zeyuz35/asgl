## YYYY-MM-DD - Optimize logistic regression canonicalization time
**Learning:** In CVXPY, using cp.multiply(y, pred) followed by cp.sum() for logistic regression objectives can be slow during canonicalization.
**Action:** Replace it with the inner product y.T @ pred to significantly speed up canonicalization and execution time.
