## 2024-05-27 - Fix pandas DataFrame feature name preservation
**Learning:** In pandas DataFrames, the `columns` attribute is a property (an instance of `pd.Index`) and not a callable method. Checking `callable(getattr(X, 'columns', None))` fails when `X` is a pandas DataFrame, resulting in silent failure to preserve feature names.
**Action:** Use `not callable(getattr(X, 'columns', None))` when checking for property-based attributes like DataFrame columns to ensure metadata preservation.
