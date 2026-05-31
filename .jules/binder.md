## 2024-05-31 - Move method-level imports to module level
**Learning:** scikit-learn tags were imported inside a method (`__sklearn_tags__`) to handle missing tags in older versions, which violates explicit namespacing and creates hidden dependencies.
**Action:** Use a module-level `try...except ImportError` block to safely import optional dependencies and keep the method bodies clean.
