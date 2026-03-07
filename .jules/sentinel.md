## 2024-05-18 - [Insecure Reflection in AdaptiveWeights]
**Vulnerability:** The codebase uses `getattr(self, "_w" + self.weight_technique)` without validating the `weight_technique` string. An attacker could set `weight_technique` to arbitrary strings, allowing them to invoke any method that starts with `_w` on the object instance.
**Learning:** Even internal library components must strictly validate parameters when using reflection (`getattr`, `eval`, etc.) to dynamically call methods based on user input, to prevent unauthorized method execution.
**Prevention:** Validate `weight_technique` against an explicit allowlist in `_check_attributes` before calling it with `getattr`.
