## 2024-05-15 - Extract sklearn tags to module level
**Learning:** scikit-learn tags imported inside functions (like `__sklearn_tags__`) can be safely extracted to the module level.
**Action:** Always wrap module-level scikit-learn imports in `try...except ImportError` blocks to ensure compatibility with older versions, and use `try...except NameError` inside functions.

## 2024-05-15 - Remove unused imports in test files
**Learning:** Unused imports in test files (like `pytest` or `numpy`) create unnecessary dependencies. Unused test variables (like `group_index` to test error messages) shouldn't be blindly deleted by sed or autofixers.
**Action:** Use targeted `# noqa: F841` statements to disable linters for specifically unused variables meant for testing purposes to prevent them from being purged or triggering linting errors.
