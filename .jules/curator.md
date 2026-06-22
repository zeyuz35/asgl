## 2025-02-28 - Fix pd.DataFrame columns attribute checking
**Learning:** Checking pandas DataFrame `columns` with `callable()` returns False and drops metadata, because it's an attribute, not a method.
**Action:** Always check `not callable(getattr(X, 'columns', None))` to properly detect property-based metadata features.
