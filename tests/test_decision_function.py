import pytest
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.datasets import make_regression, make_classification
from asgl import Regressor

def test_decision_function_lm():
    """Test decision_function for linear regression model."""
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Fit Regressor(model='lm')
    reg = Regressor(model='lm', penalization='lasso', lambda1=0.1)
    reg.fit(X, y)

    # Call decision_function(X)
    decision = reg.decision_function(X)

    # Verify output shape is (n_samples,)
    assert decision.shape == (X.shape[0],)

    # Verify output equals predict(X) for regressor
    prediction = reg.predict(X)
    np.testing.assert_allclose(decision, prediction, err_msg="decision_function output does not match predict output for lm")

    # Repeat for sparse X
    X_sparse = sparse.csr_matrix(X)
    decision_sparse = reg.decision_function(X_sparse)
    assert decision_sparse.shape == (X.shape[0],)
    np.testing.assert_allclose(decision, decision_sparse, err_msg="decision_function output mismatch between dense and sparse input for lm")

def test_decision_function_qr():
    """Test decision_function for quantile regression model."""
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

    # Fit Regressor(model='qr')
    reg = Regressor(model='qr', penalization='lasso', lambda1=0.1, quantile=0.5)
    reg.fit(X, y)

    # Call decision_function(X)
    decision = reg.decision_function(X)

    # Verify output shape is (n_samples,)
    assert decision.shape == (X.shape[0],)

    # Verify output equals predict(X) for regressor
    prediction = reg.predict(X)
    np.testing.assert_allclose(decision, prediction, err_msg="decision_function output does not match predict output for qr")

    # Repeat for sparse X
    X_sparse = sparse.csr_matrix(X)
    decision_sparse = reg.decision_function(X_sparse)
    assert decision_sparse.shape == (X.shape[0],)
    np.testing.assert_allclose(decision, decision_sparse, err_msg="decision_function output mismatch between dense and sparse input for qr")

def test_decision_function_logit():
    """Test decision_function for logistic regression model."""
    # Generate synthetic binary classification data
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)

    # Fit Regressor(model='logit')
    clf = Regressor(model='logit', penalization='lasso', lambda1=0.1)
    clf.fit(X, y)

    # Call decision_function(X)
    decision = clf.decision_function(X)

    # Verify output shape is (n_samples,)
    assert decision.shape == (X.shape[0],)

    # Verify predict_proba is consistent with decision_function output (sigmoid)
    proba = clf.predict_proba(X)
    expected_proba_pos = expit(decision)
    np.testing.assert_allclose(proba[:, 1], expected_proba_pos, err_msg="predict_proba is not consistent with decision_function for logit")

    # Verify predict is consistent with decision_function (threshold at 0)
    prediction = clf.predict(X)
    expected_prediction = (decision >= 0).astype(int)
    # The classes might be mapped differently if y wasn't 0/1, but make_classification gives 0/1.
    # Regressor implementation assumes classes are [0, 1] if y is binary 0/1.
    np.testing.assert_array_equal(prediction, clf.classes_[expected_prediction], err_msg="predict is not consistent with decision_function for logit")

    # Repeat for sparse X
    X_sparse = sparse.csr_matrix(X)
    decision_sparse = clf.decision_function(X_sparse)
    assert decision_sparse.shape == (X.shape[0],)
    np.testing.assert_allclose(decision, decision_sparse, err_msg="decision_function output mismatch between dense and sparse input for logit")
