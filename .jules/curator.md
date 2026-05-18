## 2024-05-18 - Fix feature names extraction from Pandas DataFrames
**Learning:** In `asgl.base_model.BaseModel.fit`, feature names from Pandas DataFrames are not extracted because DataFrame `columns` are `Index` properties, not methods. The code uses `callable(getattr(X, "columns", None))` which checks if `columns` is a method, not a property.
**Action:** Replace `and callable(getattr(X, "columns", None))` with `and not callable(getattr(X, "columns", None))` to correctly identify properties like `columns` in Pandas DataFrames.
