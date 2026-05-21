## 2024-05-23 - Fix pandas feature_names preservation in BaseModel.fit
**Learning:** `BaseModel.fit` incorrectly checks if `X.columns` is callable when trying to extract feature names from a pandas DataFrame. DataFrame `columns` is an Index property, not a method, so `callable(getattr(X, 'columns', None))` returns False, causing silent loss of feature names.
**Action:** Replace `callable(getattr(X, "columns", None))` with `not callable(getattr(X, "columns", None))` or simply remove the callable check so feature names are properly preserved when fitting models to DataFrames.
