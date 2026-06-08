## 2024-05-18 - Type Hint Modernization
**Learning:** Legacy `typing` module imports (`Union`, `Tuple`, `Dict`, `Optional`, `Sequence`) were widely used across the codebase despite the project requiring Python >= 3.10.
**Action:** Replaced legacy imports with PEP 585/604 native equivalents (e.g., `|` union operators, native `tuple` and `dict` types) to improve code readability and remove unnecessary imports. Also cleaned up `mosek` conflicts before running tests.
