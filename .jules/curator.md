## 2025-06-05 - Correct attribute preservation for feature names
**Learning:** When extracting feature names from inputs like pandas DataFrames in scikit-learn compatible modules, using `hasattr(X, 'columns') and callable(getattr(X, 'columns', None))` fails because `columns` is a property, not a callable method. This causes silent loss of metadata.
**Action:** Use `if hasattr(X, 'columns') and not callable(getattr(X, 'columns', None)):` to correctly distinguish pandas DataFrame columns from PySpark DataFrames (where columns is callable) and properly preserve feature names.
