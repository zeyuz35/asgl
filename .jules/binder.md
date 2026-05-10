## 2026-05-10 - Move method-level imports to module level
**Learning:** In `asgl/base_model.py`, `sklearn.utils._tags.ClassifierTags` and `sklearn.utils._tags.RegressorTags` were imported inside the `__sklearn_tags__` method definition. This violates standard Python style conventions and rules against importing libraries inside function/method definitions.
**Action:** When auditing files, search for inline imports within methods and move them to the top of the file alongside other module-level imports.
