## YYYY-MM-DD - Optimize CVXPY objective compilation for penalties
**Learning:** Replacing cp.sum(cp.multiply(...)) or cp.norm1(cp.multiply(...)) with vector inner products (@) significantly reduces CVXPY compilation and solving time for grouped/weighted penalties.
**Action:** Always prefer vector inner products (@) over element-wise multiplication and summation in CVXPY to improve performance.
