## 2024-05-18 - Type Hint Modernization
**Learning:** Python 3.10 is required by the project setup (`python_requires='>=3.10'`).
**Action:** Replace verbose legacy `typing` module imports (`Union`, `Optional`, `Tuple`, `Dict`) with native PEP 585/604 syntax (e.g., `X | Y`, `X | None`, `tuple`, `dict`) throughout the codebase.
