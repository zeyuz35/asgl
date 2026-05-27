## 2024-05-24 - Explicit scalar validation in AdaptiveWeights
**Learning:** `AdaptiveWeights` hyperparameters like `weight_tol` and `variability_pct` lack explicit input validation, which can lead to runtime errors or silent coercion later. `sklearn.utils.validation.check_scalar` is perfect for this.
**Action:** Use `check_scalar` alongside Python's `numbers.Real` to validate numeric hyperparameters within `fit_weights`.
