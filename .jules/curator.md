## 2024-05-24 - Fix feature names preservation for pandas DataFrames
**Learning:** In `asgl.base_model.BaseModel.fit`, the check `callable(getattr(X, "columns", None))` incorrectly assumes that DataFrame columns are methods. In pandas, `columns` is a property (an `Index` object), so `callable` returns False. This causes silent metadata loss, where feature names are ignored.
**Action:** Replace `callable(getattr(X, "columns", None))` with `not callable(getattr(X, "columns", None))` as described in memory, to correctly preserve pandas DataFrame feature names.
