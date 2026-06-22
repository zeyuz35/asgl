## YYYY-MM-DD - Fix DataFrame columns attribute check
**Learning:** Checking if `getattr(X, "columns", None)` is callable returns `False` for Pandas DataFrames because `.columns` is a property, leading to silent metadata loss.
**Action:** Use `not callable(...)` to properly preserve feature names from DataFrames.
