## 2024-05-22 - Move method-level imports to module level
**Learning:** Found method-level imports for `ClassifierTags` and `RegressorTags` in `asgl/base_model.py`. The codebase needs `try...except ImportError: pass` at module level and `try...except NameError: pass` in method to preserve compatibility with scikit-learn versions lacking these tags.
**Action:** When finding method-level dependency imports in scikit-learn estimators, strictly move them to the top of the file wrapped with `try...except ImportError`, and use `NameError` checks where instantiated.
