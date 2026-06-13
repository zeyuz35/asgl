## 2024-06-13 - [Pandas DataFrame Metadata Preservation]
**Learning:** Extracting feature names from pandas DataFrames using `callable(getattr(X, "columns", None))` evaluates to `False`, leading to silent metadata loss (`feature_names_in_ = None`), whereas PySpark DataFrames have callable columns.
**Action:** Use `if hasattr(X, 'columns') and not callable(getattr(X, 'columns', None)):` to appropriately identify and preserve pandas feature names.
