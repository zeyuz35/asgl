## 2024-05-27 - Fix Pandas DataFrame metadata preservation in BaseModel.fit
**Learning:** `BaseModel.fit` incorrectly checks `callable(getattr(X, "columns", None))` when trying to extract feature names from a Pandas DataFrame. `columns` is an `Index` object in Pandas (which is not callable), not a method. Because of this, `self.feature_names_in_` is set to `None` and feature names are silently lost.
**Action:** Remove the `callable()` check. Just checking `hasattr(X, "columns")` is sufficient to determine if the object has a `columns` attribute to extract feature names from.
