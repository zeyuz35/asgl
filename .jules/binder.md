## 2024-05-18 - Move Method-Level Imports to Module-Level
**Learning:** In scikit-learn compatible modules (like `asgl/base_model.py`), dynamic tags instantiation might import objects (like `ClassifierTags`) deep within method execution. This violates package hygiene and can cause runtime inefficiencies.
**Action:** Move method-level imports to the module level. To maintain backward compatibility with older library versions, wrap the module-level import in `try...except ImportError: pass` and the usage in `try...except NameError: pass`.
