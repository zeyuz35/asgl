## 2024-05-19 - Pandas feature names preservation
**Learning:** In `asgl.base_model.BaseModel.fit`, feature names from Pandas DataFrames are preserved by checking `hasattr(X, 'columns') and not callable(getattr(X, 'columns', None))` because DataFrame `columns` are `Index` properties, not methods.
**Action:** Replace `callable(getattr(X, "columns", None))` with `not callable(getattr(X, "columns", None))` to correctly preserve feature names from dataframe-like inputs.
