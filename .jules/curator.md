## 2024-05-18 - Fix feature names extraction for Pandas DataFrames
**Learning:** Pandas DataFrame `.columns` attribute is an `Index` object which is not callable. The previous check `callable(getattr(X, 'columns', None))` was incorrect and prevented capturing column names for pandas DataFrames, losing important metadata.
**Action:** Use `not callable(getattr(X, 'columns', None))` to correctly identify property-like `.columns` attributes that store metadata like feature names.
