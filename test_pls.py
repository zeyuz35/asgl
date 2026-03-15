import time
import numpy as np
from asgl.skmodels import AdaptiveWeights
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=200, n_targets=1, noise=1.0, random_state=42)
aw = AdaptiveWeights(weight_technique='pls_pct', variability_pct=0.5)

start = time.time()
for _ in range(10):
    w1 = aw._wpls_pct(X, y)
t1 = time.time()
print("Time taken by optimized _wpls_pct (10 iterations):", t1 - start)
