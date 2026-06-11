## YYYY-MM-DD - Type hint modernization
**Learning:** Python 3.10+ projects allow the use of PEP 585/604 type hints (e.g., `X | Y`, `tuple`, `dict`) natively, making `typing.Union`, `typing.Optional`, `typing.Tuple`, and `typing.Dict` obsolete and verbose. The `asgl` project requires python >= 3.10.
**Action:** Replace legacy `typing` imports with modern native equivalents in `constants.py`, `utils.py`, `regressor.py`, `adaptive_weights.py`, and `base_model.py`.
