## 2026-05-26 - Move method-level scikit-learn tags imports to module level
**Learning:** scikit-learn `ClassifierTags` and `RegressorTags` should be imported at the module level inside a `try...except ImportError` block to support older scikit-learn versions without these tags, preventing method-level import issues.
**Action:** Move `ClassifierTags` and `RegressorTags` to the module level and use `try...except ImportError` in scikit-learn compatible modules.
