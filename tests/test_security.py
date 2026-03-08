import pytest
import numpy as np
from asgl.skmodels import Regressor

def test_weight_technique_validation():
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    # Test that an invalid weight_technique raises a ValueError
    # and prevents insecure reflection via getattr.
    model = Regressor(penalization='alasso', weight_technique='invalid_technique')
    with pytest.raises(ValueError, match="weight_technique must be one of"):
        model.fit(X, y)

def test_weight_technique_valid():
    X = np.random.randn(10, 5)
    y = np.random.randn(10)

    # Test that a valid weight_technique completes successfully.
    model = Regressor(penalization='alasso', weight_technique='ridge')
    model.fit(X, y) # Should pass without ValueError
