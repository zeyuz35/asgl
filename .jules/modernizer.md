## 2024-06-18 - Type Hints Modernization (PEP 585/604)
**Learning:** Python >= 3.10 allows replacing legacy `typing` constructs (`Union`, `Optional`, `Tuple`, `Dict`) with native types (`|`, `| None`, `tuple`, `dict`) and `Sequence` should come from `collections.abc` instead of `typing`. The codebase relies on typing across all files.
**Action:** Replace `from typing import ...` with native types and move `Sequence` to `from collections.abc import Sequence` to adhere to modern standards.
