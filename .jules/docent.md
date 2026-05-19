## 2024-05-19 - Add missing public API docstrings and fix typos
**Learning:** The public API methods (`fit`, `predict`, `predict_proba`, `decision_function`, `score`) in `BaseModel` and `Regressor` lack docstrings, which degrades user experience and documentation completeness. Additionally, there are typos in parameter docstrings like `defaul=0.1`.
**Action:** When working on sklearn-compatible estimators, always ensure that standard methods like `fit` and `predict` have complete numpydoc-style docstrings describing parameters and return values.
