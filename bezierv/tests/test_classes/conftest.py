import pytest
import numpy as np
from bezierv.classes.bezierv import Bezierv

@pytest.fixture
def normal_data(scope='package') -> np.array:
    """
    Fixture to create a sample data instance for testing.
    
    Returns
    -------
    np.array
        A numpy array of sample data points.
    """
    np.random.seed(111)
    return np.random.normal(loc=0, scale=1, size=100)

@pytest.fixture
def linear_bezierv() -> Bezierv:
    """
    Fixture to create a linear Bezier instance for testing.
    
    Returns
    -------
    Bezierv
        An instance of the Bezierv class with linear controls.
    """
    return Bezierv(n=1, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))

@pytest.fixture
def two_uniform_bezierv() -> tuple:
    """
    Fixture to create two uniform Bezier random variables for testing convolution.
    
    Returns
    -------
    tuple
    """
    return (
        Bezierv(n=1, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0])),
        Bezierv(n=1, controls_x=np.array([0.0, 1.0]), controls_z=np.array([0.0, 1.0]))
    )