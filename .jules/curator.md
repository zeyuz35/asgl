## 2024-06-01 - Fix silent metadata loss for Pandas DataFrames
**Learning:** When extracting feature names from inputs like pandas DataFrames, use `not callable(getattr(X, 'columns', None))` to correctly distinguish them from PySpark DataFrames and prevent silent metadata loss.
**Action:** Always verify `hasattr(X, 'columns') and not callable(getattr(X, 'columns', None))` when preserving DataFrame feature names.
