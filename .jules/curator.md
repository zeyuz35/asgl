## YYYY-MM-DD - Fix feature names extraction from pandas DataFrame
**Learning:** When extracting feature names from pandas DataFrames, ensure the check verifies the attribute is not callable. The `.columns` attribute is a property, and checking it as a callable will evaluate to `False` and cause silent metadata loss.
**Action:** Always check that properties are not callable when trying to extract metadata like DataFrame columns.
