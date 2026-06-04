## 2024-05-24 - Fix feature names extraction for pandas DataFrames
**Learning:** The check `callable(getattr(X, "columns", None))` incorrectly fails for pandas DataFrames where `columns` is a property, leading to silent metadata loss.
**Action:** Use `not callable(getattr(X, "columns", None))` to distinguish properties like pandas columns from callable methods.
