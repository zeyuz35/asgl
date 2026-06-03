## 2025-06-03 - Move sklearn tags to module level
**Learning:** scikit-learn tags can cause runtime crashes on older sklearn versions if imported at the method level. Using a module-level `try...except ImportError` handles this gracefully without sacrificing readability.
**Action:** Always prefer importing conditional classes at the module level using `try...except ImportError` with `pass` to keep method scopes clean and maintain backwards compatibility.
