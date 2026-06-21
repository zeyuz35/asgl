## 2024-06-21 - Move scikit-learn tags imports to module level
**Learning:** scikit-learn tags (`ClassifierTags`, `RegressorTags`) were imported inside the `__sklearn_tags__` method definition.
**Action:** Moving conditional backward-compatibility imports out of method definitions to the module level and wrapping them in a `try...except ImportError` block improves structural hygiene and follows best practices while preserving original functionality.
