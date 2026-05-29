## 2024-05-24 - Pandas DataFrame feature names preservation
**Learning:** Pandas DataFrame `columns` are `Index` properties, not methods, meaning `callable(getattr(X, 'columns', None))` returns False.
**Action:** When validating structured data objects like DataFrames, ensure that property-based attributes are checked with `not callable()` to avoid dropping metadata.
