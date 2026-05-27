## 2024-05-27 - Move method-level imports to module-level
**Learning:** scikit-learn tags (`ClassifierTags`, `RegressorTags`) were imported inside the `__sklearn_tags__` method to maintain compatibility with older `scikit-learn` versions. Moving them to the module-level blindly would break tests.
**Action:** Always wrap module-level imports of newer external library features in `try...except ImportError` blocks and check against `None` at the call site to maintain backward compatibility while adhering to the rule against method-level imports.
