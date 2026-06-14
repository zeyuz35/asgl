## 2024-06-14 - Fix pandas feature names extraction
**Learning:** In pandas DataFrames, the columns attribute is a property, not a callable method. Checking callable(getattr(X, 'columns', None)) will return False, leading to silent metadata loss when extracting feature names.
**Action:** Use not callable(getattr(X, 'columns', None)) to properly verify that the attribute is a data property rather than a method before accessing it.
