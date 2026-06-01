## 2024-05-24 - Module Level Imports for sklearn tags
**Learning:** scikit-learn tags (`ClassifierTags`, `RegressorTags`) should be imported at the module level within a `try...except ImportError` block to support older versions and avoid importing inside functions.
**Action:** Always move method-level imports to the module level.
