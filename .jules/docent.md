## 2024-05-24 - ASCII Only Source and Documentation
**Learning:** Found non-ASCII characters like em dashes (`—`), Greek letters (`λ`), ellipses (`…`), superscript two (`²`), and arrows (`→`) in both `README.md` and test files (`tests/test_skmodels.py`, `tests/test_skmodels_sparse.py`). These violate the ASCII-only directive.
**Action:** Always replace non-ASCII characters with their ASCII equivalents (e.g., `-`, `lambda1`, `...`, `^2` or `squared`, `->`) across source code, documentation, and commit messages to ensure universal portability.
