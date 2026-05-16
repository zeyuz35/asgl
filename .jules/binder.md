## 2026-05-16 - [Method-Level Imports]
**Learning:** Importing within methods violates best practices and can hide dependencies. However, moving them to module level requires careful handling of backward compatibility for libraries like scikit-learn that might introduce new features (like Tags) in recent versions.
**Action:** Move method-level imports to module level. Wrap in try...except ImportError to maintain backward compatibility, and use try...except NameError where instantiated.
