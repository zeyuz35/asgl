import pytest
import numpy as np
import cvxpy as cp
from asgl import Regressor
from sklearn.datasets import make_regression


def generate_data(n_samples=100, n_features=10, noise=0.1, random_state=42):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    return X, y


def test_solver_fallback_with_invalid_solver():
    X, y = generate_data()

    # "ASD" is an invalid solver name and should fail validation
    solvers = ["ASD", "OSQP", "SCS", "CLARABEL"]

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver=solvers)

    # We expect a ValueError due to invalid solver
    with pytest.raises(ValueError, match="Invalid solver 'ASD'"):
        model.fit(X, y)


def test_solver_fallback_all_requested_fail():
    X, y = generate_data()

    # Use a valid solver that we know isn't installed or mock an error
    # For testing fallback, let's use a valid solver that fails.
    # We will simulate failure by mocking cp.Problem.solve to raise an error for a valid solver
    solvers = ["OSQP"]

    # We will mock the Problem.solve method to fail when OSQP is called
    original_solve = cp.Problem.solve

    def mock_solve(self, *args, **kwargs):
        if kwargs.get("solver") == "OSQP":
            raise ValueError("Simulated OSQP failure")
        return original_solve(self, *args, **kwargs)

    model = Regressor(model="lm", penalization="lasso", lambda1=0.1, solver=solvers)

    import unittest.mock
    with unittest.mock.patch.object(cp.Problem, 'solve', new=mock_solve):
        with pytest.warns(RuntimeWarning) as record:
            model.fit(X, y)

    assert model.is_fitted_
    # It should have fallen back to an installed solver
    assert model.solver_stats_["solver_name"] in cp.installed_solvers()
    assert model.solver_stats_["solver_name"] != "OSQP"

    # Check for the sequence of warnings
    warnings_list = [str(w.message) for w in record]

    # 1. OSQP failed
    assert any("Solver OSQP failed" in w for w in warnings_list)

    # 2. All requested failed, trying remaining
    assert any(
        "Requested solver(s) ['OSQP'] failed. Trying remaining installed solvers" in w
        for w in warnings_list
    )

    # 3. Successfully solved with fallback
    assert any("Successfully solved with fallback solver" in w for w in warnings_list)
