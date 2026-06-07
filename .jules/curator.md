## YYYY-MM-DD - Fix Pandas DataFrame metadata loss
**Learning:** `hasattr(X, "columns")` is necessary, but checking `callable(getattr(X, "columns", None))` evaluates to true in PySpark and false in Pandas, leading to silent metadata loss for Pandas inputs.
**Action:** Use `if hasattr(X, "columns") and not callable(getattr(X, "columns", None)):` to accurately detect pandas DataFrame features without losing metadata.
