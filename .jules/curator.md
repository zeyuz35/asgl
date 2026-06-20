## YYYY-MM-DD - [Fix loss of pandas feature names]
**Learning:** Checking for pandas DataFrame attributes using `callable(getattr(X, 'columns', None))` returns `False` because `.columns` is a property, leading to silent loss of metadata (`feature_names_in_`).
**Action:** Always verify `not callable(getattr(X, 'columns', None))` when safely checking for a property vs method.
