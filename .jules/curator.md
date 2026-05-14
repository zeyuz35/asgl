## 2024-05-14 - Fix Feature Names Preservation Bug
**Learning:** `callable(getattr(X, 'columns', None))` fails silently for Pandas DataFrames because `columns` is a Pandas `Index` property, not a callable method, resulting in `feature_names_in_` never being preserved.
**Action:** Use `hasattr(X, 'columns')` to verify the presence of column metadata without checking if it is callable. When testing Pandas interaction without requiring Pandas as a dependency, use a mock object returning an array from `__array__` with shape, ndim, and the required `columns` attribute.
