## YYYY-MM-DD - Prevent silent metadata loss for pandas DataFrames
**Learning:** In sklearn-compatible estimators, when extracting feature names from pandas DataFrames, use `hasattr(X, 'columns') and not callable(getattr(X, 'columns', None))` to correctly distinguish them and prevent silent metadata loss.
**Action:** Use correct attribute validation to preserve data integrity.
