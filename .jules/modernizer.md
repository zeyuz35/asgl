## YYYY-MM-DD - Type hint modernization
**Learning:** In Python 3.10+ projects, a great modernization strategy is to update type hints from legacy `typing` module imports (e.g. `Union`, `Optional`, `Tuple`, `Dict`) to their native equivalents (e.g. `X | Y`, `X | None`, `tuple`, `dict`), which reduces dependency bloat and improves clarity.
**Action:** Use native type aliases and the bitwise OR operator `|` for unions in the future.
