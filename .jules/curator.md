## YYYY-MM-DD - Fix pandas DataFrame columns metadata preservation
**Learning:** Extracting pandas .columns attribute while checking if it is callable evaluates to False because .columns is a property, causing silent metadata loss.
**Action:** Always check not callable(getattr(X, "columns", None)) when preserving DataFrame feature names.
