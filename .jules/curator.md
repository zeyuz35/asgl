## YYYY-MM-DD - Fix pandas dataframe feature names extraction
**Learning:** In sklearn-compatible estimators, when extracting feature names from inputs like pandas DataFrames, use `if hasattr(X, 'columns') and not callable(getattr(X, 'columns', None)):` to correctly distinguish them from PySpark DataFrames (where `columns` is callable) and prevent silent metadata loss.
**Action:** Replace `callable(getattr(X, 'columns', None))` with `not callable(getattr(X, 'columns', None))` in `asgl/base_model.py`.
