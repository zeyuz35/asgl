
import numpy as np
from asgl import Regressor
import pytest

def test_adaptive_weights_recalculation():
    """
    Verify that adaptive weights are recalculated on subsequent fit calls
    and do not persist from previous fits.
    """
    # Create two different datasets
    # Dataset 1: Feature 0 is important, Feature 1 is noise
    X1 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    y1 = np.array([1, 1, 0, 0], dtype=float)
    group_index = [1, 2] # 2 features, 2 groups

    # Dataset 2: Feature 1 is important, Feature 0 is noise
    # We essentially flip the relationship
    X2 = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
    y2 = np.array([1, 1, 0, 0], dtype=float)

    # Use unpenalized weight technique for deterministic behavior
    model = Regressor(
        model='lm',
        penalization='agl',
        lambda1=0.1,
        weight_technique='unpenalized',
        solver='CLARABEL'
    )

    # First fit
    model.fit(X1, y1, group_index=group_index)
    weights1 = model.group_weights_.copy()

    # Second fit on different data
    model.fit(X2, y2, group_index=group_index)
    weights2 = model.group_weights_.copy()

    # Assert that weights are different
    # Since X1 favors feature 0 and X2 favors feature 1, the weights should be roughly flipped
    assert not np.allclose(weights1, weights2), "Weights should differ between fits on different data"

    # Detailed check:
    # In X1, feat 0 coef ~ 1, feat 1 coef ~ 0. Weight ~ 1/|coef|^p
    # So weight for feat 0 should be LOW, weight for feat 1 should be HIGH
    assert weights1[0] < weights1[1], "For X1, group 1 (feat 0) should have lower weight than group 2 (feat 1)"

    # In X2, feat 0 coef ~ 0, feat 1 coef ~ 1.
    # So weight for feat 0 should be HIGH, weight for feat 1 should be LOW
    assert weights2[0] > weights2[1], "For X2, group 1 (feat 0) should have higher weight than group 2 (feat 1)"

if __name__ == "__main__":
    test_adaptive_weights_recalculation()
