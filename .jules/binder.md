## 2025-02-28 - Move inner scikit-learn imports to module level
**Learning:** scikit-learn tags classes like `ClassifierTags` and `RegressorTags` were imported dynamically inside `__sklearn_tags__`. It is cleaner and reduces dependency overhead per method call to attempt these imports at the module level.
**Action:** Move inner imports to the top of the file, wrapped in `try...except ImportError` to safely support older scikit-learn versions without breaking functionality.
