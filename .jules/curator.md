## 2025-03-09 - Fix metadata loss for pandas DataFrame inputs
**Learning:** In sklearn-compatible estimators, when extracting feature names from inputs like pandas DataFrames, `columns` is a property, not a callable.
**Action:** Use `if hasattr(X, 'columns') and not callable(getattr(X, 'columns', None)):` to correctly distinguish them from PySpark DataFrames (where `columns` is callable) and prevent silent metadata loss.
