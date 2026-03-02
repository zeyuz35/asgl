## 2024-03-02 - [Reflection Vulnerability via Unvalidated getattr]
**Vulnerability:** Found unvalidated user input (`weight_technique`) being used directly inside a `getattr` call (`getattr(self, "_w" + self.weight_technique)`) within `AdaptiveWeights.fit_weights`.
**Learning:** This is a reflection vulnerability where users could potentially instantiate objects with malicious techniques to execute unintended methods matching the `_w` prefix. Subclasses must rigorously validate dynamic attributes.
**Prevention:** Implement input validation for variables driving `getattr`. Maintain an explicitly defined list of allowed values (e.g., `ALLOWED_WEIGHT_TECHNIQUES`) and assert the parameter exists in this subset.
