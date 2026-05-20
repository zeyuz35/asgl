## 2024-05-20 - Pandas DataFrame column preservation
**Learning:** Pandas DataFrame `.columns` attribute is an `Index` property, not a callable method. Checking `callable(getattr(X, "columns", None))` evaluates to `False`, leading to silent loss of column names during model fitting.
**Action:** Use `not callable(getattr(X, "columns", None))` when checking if an object exposes a dataframe-like columns property to correctly preserve feature names without assuming a specific type.
