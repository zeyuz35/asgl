## YYYY-MM-DD - Fix empty f-string in tests
**Learning:** Found an unnecessary f-string without placeholders in `tests/test_skmodels.py`, which is a style violation and caught by Ruff.
**Action:** Remove the `f` prefix using `replace_with_git_merge_diff`.
