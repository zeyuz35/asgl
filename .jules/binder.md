## 2025-05-10 - Import Libraries Inside Method Definitions
**Learning:** `ClassifierTags` and `RegressorTags` from `sklearn.utils._tags` were imported locally inside the `__sklearn_tags__` method definition of `asgl/base_model.py`, violating the rule against importing libraries inside function/method definitions.
**Action:** Move local imports to module level at the top of the file to adhere to PEP 8 standards and ensure dependencies are explicitly declared up-front.
