## YYYY-MM-DD - Move scikit-learn inner imports to top level
**Learning:** When removing inner `scikit-learn` imports (e.g., `ClassifierTags`, `RegressorTags`) from the `__sklearn_tags__` method, be careful to only remove the import statements themselves and preserve the required adjacent functional assignments. Move the imports to a module-level `try...except ImportError` block with `pass` in the except block.
**Action:** Use a top level try-except block to optionally import classes for backward compatibility while keeping function body clean.
