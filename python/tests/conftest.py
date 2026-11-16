import numpy as np
import pytest


@pytest.fixture
def model():
    def feature(t, tau):
        return np.array([1.]) if tau == 0 else np.array([1., 1.])

    def prior(t):
        return ({"mu": np.zeros(1), "sigma": np.eye(1)} if t == 1 else
                {"mu": np.zeros(2), "sigma": np.array([[1., -1.], [-1., 2.]])})

    def transform(tau):
        return np.array([[1.]]) if tau == 0 else np.array([[1., 1.]])

    return feature, prior, transform
