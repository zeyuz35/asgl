## 2024-03-01 - Optimizing PCA for Sparse Matrices
**Learning:** scikit-learn's `PCA(n_components="some_fraction")` does not natively support sparse matrices because `svd_solver='full'` requires dense conversions. When `AdaptiveWeights` tries to achieve `variability_pct < 1` for sparse data by iteratively incrementing `n_comp` from `1` to `max_comp` and re-running PCA, it implicitly triggers a dense conversion on every loop iteration, leading to massive memory and performance overhead (23s+ for small inputs).
**Action:** Replace iterative PCA incrementing with a single run of `PCA(n_components=max_comp, svd_solver="arpack")` to calculate all principal components at once without iterative dense conversions. Then compute the cumulative explained variance manually to truncate components.

## 2025-02-18 - Avoid np.bincount on unstructured group indices
**Learning:** `np.bincount` is memory-inefficient and prone to crashing when dealing with potentially sparse or negatively-indexed unstructured data groups (it creates an array up to the max index size).
**Action:** Use `np.argsort` coupled with `np.add.reduceat` to sum elements grouped by index, ensuring O(N log N) time and linear memory space, completely independent of the maximum value of the group labels.
