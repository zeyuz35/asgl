## 2025-06-09 - Fix DataFrame Feature Name Extraction Logic
**Learning:** In sklearn-compatible estimators, when extracting feature names from inputs like pandas DataFrames, hasattr(X, 'columns') should be paired with not callable(getattr(X, 'columns', None)) to correctly distinguish them from PySpark DataFrames and prevent silent metadata loss.
**Action:** Always verify if columns attribute is callable when extracting feature names from DataFrame-like structures to avoid type instability and metadata preservation issues.
