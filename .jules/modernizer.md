## YYYY-MM-DD - Modernize legacy type hints to PEP 585/604 native syntax
**Learning:** Python >= 3.10 allows for cleaner native type hints via the bitwise OR operator and built-in type mapping eliminating the need for typing imports like Union, Optional, Tuple, Dict, and Sequence.
**Action:** Replace all typing module type hint annotations with PEP 585/604 alternatives to improve maintainability and modernize the codebase without altering functionality.
