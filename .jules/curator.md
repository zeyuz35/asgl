## 2024-05-30 - Fix DataFrame columns attribute preservation
**Learning:** Pandas DataFrame `.columns` is a property, not a callable. Checking `callable(getattr(X, 'columns', None))` wrongly drops column names.
**Action:** Use `not callable(getattr(X, 'columns', None))` to properly detect and preserve dataframe column names as feature names.
