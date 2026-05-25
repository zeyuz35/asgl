## 2024-05-25 - Typo in parameter docstring in asgl/regressor.py
**Learning:** Found `lambda1: float, defaul=0.1` typo in docstring. Also need to ensure uniform parameter formatting. There are missing spaces around `=` for `default` in some docstrings (`model: str, default = 'lm'`, `penalization: str or None, default = 'lasso'`) vs `quantile: float, default=0.5`.
**Action:** Fix typos and unify formatting of `default=...` in docstrings.
