## 2024-05-28 - Move method-level imports to module level in asgl/base_model.py
**Learning:** Found scikit-learn tags imported inside a method, which violates the rule against importing libraries inside functions.
**Action:** Always move imports to the module level, wrapping them in try/except if necessary to maintain compatibility.
