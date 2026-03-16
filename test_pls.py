import numpy as np
import asgl
from sklearn.datasets import make_regression
from asgl import Regressor

X, y = make_regression(n_samples=100, n_features=10, random_state=42)

model = Regressor(model='lm', penalization='alasso', weight_technique='pls_pct', variability_pct=0.9)
model.fit(X, y)
print("pls_pct individual weights:\n", model.individual_weights_)
