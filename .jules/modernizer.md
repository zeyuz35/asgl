## 2025-06-09 - Type Hinting Modernization
**Learning:** The project uses Python >= 3.10, making it eligible for PEP 585/604 native type hinting (e.g., `X | Y` instead of `Union[X, Y]`, `tuple` instead of `Tuple`).
**Action:** Replace all legacy typing imports (`Union`, `Optional`, `Tuple`, `Dict`) with modern native equivalents to improve code cleanliness and reduce unnecessary imports.
