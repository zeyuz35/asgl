# Binder's Journal

## 2024-05-19 - Removed in-function imports
**Learning:** When removing seemingly unused variables (e.g., to fix `F841` linting errors), it's crucial to verify the variable is not used further down in the file or used across other test methods. A global `sed` replacement adding `# noqa: F841` to a common variable name like `group_index` across hundreds of tests can be sloppy.
**Action:** Instead of blind global replacements, remove only the explicitly unused instances (e.g., by deleting the variable assignment entirely in the specific test method) after confirming it is completely unused.
