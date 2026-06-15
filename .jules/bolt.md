## YYYY-MM-DD - Inner product optimization for CVXPY logit objective
**Learning:** Using `cp.multiply` for element-wise multiplication followed by `cp.sum` in CVXPY `logit` objectives scales poorly for large arrays during canonicalization and solve time.
**Action:** Replace `cp.sum(cp.logistic(pred_flat) - cp.multiply(y_flat, pred_flat))` with `cp.sum(cp.logistic(pred_flat)) - y_flat.T @ pred_flat` to leverage efficient matrix multiplication for the inner product.
