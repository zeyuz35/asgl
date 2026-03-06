## 2024-05-21 - [Fix insecure reflection in weight technique getattr]
**Vulnerability:** In `asgl/skmodels.py`, the `weight_technique` string was being directly concatenated to `"_w"` and passed into `getattr` without explicit validation, which could have allowed arbitrary execution of internal class methods.
**Learning:** This vulnerability existed due to missing validation on the `weight_technique` string parameter and the implicit trust in user input when retrieving methods with reflection via `getattr`.
**Prevention:** Always validate user input parameters using strict type checking (e.g., `isinstance(val, str)`) and explicit allowlist membership testing before passing the values to `getattr`, `eval`, or similarly powerful reflection or execution utilities.
