## 2025-02-21 - Modernize typing imports to PEP 585/604 native syntax
**Learning:** Legacy `typing` imports like `Union`, `Optional`, `Tuple`, and `Dict` can be successfully modernized to native constructs (`|`, `tuple`, `dict`) in codebases requiring Python 3.10+. Abstract collections like `Sequence` should be imported from `collections.abc` instead of `typing`.
**Action:** Always prefer native syntax when Python requirements allow it to reduce `typing` boilerplate.
