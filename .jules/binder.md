## 2025-02-12 - Moved inner scikit-learn tags imports to module level
**Learning:** Inner function/method imports can clutter the code and cause subtle issues. They should be moved to the module level if possible, wrapping them in `try...except ImportError` blocks if they might not be available in all supported versions of a dependency (like older scikit-learn versions).
**Action:** Always scan for inner imports inside methods or functions and move them to the module-level namespace to keep dependency structures clean and explicit.
