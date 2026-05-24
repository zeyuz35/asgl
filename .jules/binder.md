## 2024-05-24 - Module Level Imports
**Learning:** scikit-learn tags (`ClassifierTags`, `RegressorTags`) were being imported inside methods, which violates packaging best practices.
**Action:** Move these imports to the module level wrapped in `try...except ImportError: pass` blocks and handle missing variables via `try...except NameError: pass` inside the methods to ensure backwards compatibility with older versions of `scikit-learn`.
