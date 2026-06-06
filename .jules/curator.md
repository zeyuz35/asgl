## 2024-05-24 - Fix silent metadata loss for feature names
**Learning:** The logic distinguishing pandas DataFrame structures (`columns` is an attribute) from PySpark DataFrames (`columns` is callable) was inverted, leading to silent metadata loss for standard pandas DataFrames during input validation.
**Action:** Always verify that the property introspection matches the actual attribute semantics of the target framework (e.g., pandas DataFrame `columns` attribute should not be callable).
