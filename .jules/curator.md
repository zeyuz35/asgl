## 2024-06-18 - Fix silent metadata loss when capturing feature names
**Learning:** `getattr(X, 'columns')` on a pandas DataFrame returns a property which is not callable. Checking it as callable (`callable(getattr(X, 'columns', None))`) evaluates to `False` and causes silent metadata loss.
**Action:** Always check that property attributes like `columns` are `not callable` to ensure they are valid data attributes and not functions.
