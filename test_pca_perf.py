import numpy as np
import time
from sklearn.decomposition import PCA

X = np.random.randn(1000, 100) # (samples, features)

start = time.time()
for _ in range(100):
    max_comp = np.min(X.shape) - 1 # 99
    pca = PCA(n_components=max_comp, svd_solver="arpack")
    pca.fit(X)
print("arpack:", time.time() - start)

start = time.time()
for _ in range(100):
    max_comp = np.min(X.shape) - 1 # 99
    pca = PCA(n_components=max_comp, svd_solver="auto")
    pca.fit(X)
print("auto:", time.time() - start)
