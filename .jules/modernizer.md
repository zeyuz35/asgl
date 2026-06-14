## 2025-02-26 - Update type hints to PEP 585/604
**Learning:** Legacy `typing` module imports (`Sequence`, `Optional`, `Tuple`, `Union`, `Dict`) can be replaced with PEP 585/604 native equivalents (e.g., `X | Y`, `X | None`, `tuple`, `dict`) since python requires `>=3.10`.
**Action:** Replace these in the codebase when working as the Modernizer agent.
