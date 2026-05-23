## 2024-05-15 - Move scikit-learn tags imports to module level
**Learning:** scikit-learn's `ClassifierTags` and `RegressorTags` are imported locally inside `__sklearn_tags__` which violates Binder guidelines. To keep compatibility, they must be imported with a try-except at the top of the module.
**Action:** Always move local module imports to the top level unless lazy loading is explicitly required for significant performance gains or to break dependency circles. Ensure backward compatibility with older libraries by wrapping imports in `try...except ImportError` blocks when necessary.
