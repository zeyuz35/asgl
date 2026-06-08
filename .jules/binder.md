## 2024-06-08 - Move scikit-learn tag imports to module level
**Learning:** Inner imports inside methods can cause problems and it is cleaner to keep them at the module level, wrapping them in try/except blocks to preserve compatibility with optional or newer dependencies.
**Action:** Move inner imports to module level using try/except blocks to preserve compatibility and maintain clean namespaces.
