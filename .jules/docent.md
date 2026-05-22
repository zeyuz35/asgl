## 2025-05-22 - Fix default parameter formatting and typo in docstring
**Learning:** Some models had spaces around the equals sign in their default parameter format (e.g. `default = 'lm'`) and typos like `defaul=0.1`.
**Action:** Use regex `default *=` or `defaul=` to spot inconsistencies and standardize to `default='value'`.
