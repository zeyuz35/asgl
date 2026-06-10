## 2025-02-13 - Move sklearn tags imports to module level
**Learning:** Method-level imports of scikit-learn tags (ClassifierTags, RegressorTags) in `__sklearn_tags__` can be moved to the module level.
**Action:** Use a `try...except ImportError: pass` block at the module level to handle older versions of scikit-learn while keeping the `__sklearn_tags__` logic clean and avoiding inner imports.
