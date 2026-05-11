## 2024-05-24 - Avoid importing in method definition
**Learning:** Found an `import` statement for `ClassifierTags` and `RegressorTags` within the `__sklearn_tags__` method. This violates the "Never import libraries inside function/method definitions" rule.
**Action:** Move the `import` statements to the module level and fix the usage.
