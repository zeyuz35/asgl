## 2024-05-23 - Pandas DataFrame column attribute check
**Learning:** Pandas DataFrame columns are `Index` properties, not methods, meaning `callable(getattr(X, 'columns', None))` returns False.
**Action:** When extracting feature names from potential DataFrames, ensure we check for `hasattr(X, 'columns') and not callable(getattr(X, 'columns', None))` to correctly preserve metadata.
