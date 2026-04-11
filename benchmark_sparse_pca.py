import numpy as np
import time
from sklearn.decomposition import PCA

X = np.random.randn(10000, 100)

start = time.time()
for _ in range(10):
    pca = PCA(n_components=99, svd_solver="full")
    pca.fit(X)
print("full:", time.time() - start)

start = time.time()
for _ in range(10):
    pca = PCA(n_components=99, svd_solver="arpack")
    pca.fit(X)
print("arpack:", time.time() - start)

start = time.time()
for _ in range(10):
    pca = PCA(n_components=99, svd_solver="auto")
    pca.fit(X)
print("auto:", time.time() - start)
