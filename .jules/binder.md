## YYYY-MM-DD - Moved inner scikit-learn imports to module level
**Learning:** Inner imports inside method definitions can hide dependencies and reduce codebase hygiene. Older scikit-learn versions may not support ClassifierTags or RegressorTags, so handling them at the module level with a try-except block is more robust.
**Action:** Move method-level imports to the top module level when possible, utilizing fallback mechanisms like `try...except ImportError` if cross-version compatibility is needed.
