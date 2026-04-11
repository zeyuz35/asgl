import numpy as np
import cvxpy as cp
import time
from scipy import sparse

X = np.random.randn(100, 10)
y = np.random.randn(100, 1)

beta = cp.Variable((10, 1))

start = time.time()
for _ in range(100):
    pred = X @ beta
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - pred)))
    prob.solve()
print("X @ beta (numpy):", time.time() - start)

start = time.time()
for _ in range(100):
    X_const = cp.Constant(X)
    pred = X_const @ beta
    prob = cp.Problem(cp.Minimize(cp.sum_squares(y - pred)))
    prob.solve()
print("cp.Constant(X) @ beta:", time.time() - start)
