## 2025-02-26 - Fix silent loss of feature names for pandas DataFrames
**Learning:** In pandas, `.columns` is a property (Index object), not a method. Checking if it's callable `callable(getattr(X, 'columns', None))` will incorrectly return False, causing silent metadata loss for DataFrames.
**Action:** When extracting feature names from pandas DataFrames, ensure the check explicitly verifies the attribute is not callable: `not callable(getattr(X, 'columns', None))` to preserve domain-specific metadata.
