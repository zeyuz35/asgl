## 2024-05-23 - Module Level Imports
**Learning:** scikit-learn tags (ClassifierTags, RegressorTags) imported inside methods can cause dependency and structure issues, violating the rule against importing libraries inside method definitions.
**Action:** Move these imports to the module level inside a try...except ImportError block using pass in the except block to preserve functionality and backward compatibility.
