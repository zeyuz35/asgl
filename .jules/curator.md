## 2024-05-25 - [Fix feature_names_in_ preservation for pandas DataFrames]
**Learning:** In `asgl.base_model.BaseModel.fit`, feature names from Pandas DataFrames are not preserved because the condition checks `callable(getattr(X, 'columns', None))`. Pandas `DataFrame.columns` is an `Index` property, not a method, so this evaluates to `False`.
**Action:** Remove the `callable` check. Just check if `hasattr(X, 'columns') and not callable(getattr(X, 'columns', None))` or just check if it's an iterable and we can convert it to numpy array.
