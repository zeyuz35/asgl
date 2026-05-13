## 2024-05-24 - Fix DataFrame Feature Names Extraction
**Learning:** `callable(getattr(X, "columns", None))` fails for pandas DataFrames because `.columns` is a property returning an `Index` object, which is not callable.
**Action:** Use `hasattr(X, "columns")` instead of `callable(...)` when checking if an object has a `columns` attribute to extract feature names.
