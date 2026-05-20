## 2024-05-18 - Optimize CVXPY Group Norms Multiplication
**Learning:** CVXPY canonicalization scales poorly when using element-wise multiplication (`cp.multiply`) followed by a sum over group norms (`cp.sum(cp.multiply(weights, group_norms))`).
**Action:** Replace `cp.sum(cp.multiply(a, b))` with the more efficient, native matrix-vector inner product `a @ b` which is functionally equivalent, dramatically faster during canonicalization, and natively supported by CVXPY while being more Pythonic.

## 2024-05-18 - Logistic Objective Benchmarking
**Learning:** Replacing `cp.sum(cp.logistic(pred) - cp.multiply(y, pred))` with the vector inner product `cp.sum(cp.logistic(pred)) - y @ pred` in CVXPY can degrade actual solver execution time for matrix variables (like multi-output), even if canonicalization time is identical.
**Action:** Always benchmark both canonicalization time AND actual solve time before replacing legacy objective formulations, especially when non-linear functions like `cp.logistic` are involved.
