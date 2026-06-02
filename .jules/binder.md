## 2024-05-15 - Initial\n**Learning:** Nothing yet.\n**Action:** Nothing yet.
## 2024-05-15 - Move scikit-learn tags import to module level
**Learning:** Moving scikit-learn tags (ClassifierTags, RegressorTags) imports to the module level in a `try...except ImportError` block improves code hygiene by removing inner imports inside methods, while preserving backwards compatibility. The `except` block should use `pass` to avoid resetting objects that are conditionally defined.
**Action:** When modernizing scikit-learn inner imports for compatibility tags, move them to a top-level try/except block.
