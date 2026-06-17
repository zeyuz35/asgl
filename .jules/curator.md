## YYYY-MM-DD - Fix pandas feature names extraction
**Learning:** Checking pandas DataFrame .columns property as callable prevents proper extraction of feature names because it is an Index object, not a method.
**Action:** Always check that property-like metadata attributes are not callable before accessing them to avoid silent metadata loss.
