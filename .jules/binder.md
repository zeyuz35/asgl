## 2024-05-29 - Move method-level scikit-learn tag imports to module level
**Learning:** scikit-learn tags (`ClassifierTags`, `RegressorTags`) were imported inside `__sklearn_tags__`, violating the rule against method-level imports. Older scikit-learn versions lack these tags, which can cause `ImportError`.
**Action:** Always place external library imports at the module level. Use `try...except ImportError` blocks for optional or version-dependent scikit-learn features to ensure compatibility with older environments.
