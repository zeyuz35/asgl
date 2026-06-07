## YYYY-MM-DD - Move scikit-learn tags imports to module level
**Learning:** In sklearn-compatible estimators, inner imports within methods can cause repeated module loading or performance hits. They should be moved to the module level.
**Action:** Use a try...except ImportError: pass block at the top of the file to import sklearn tags classes, maintaining backward compatibility while avoiding inner imports.
