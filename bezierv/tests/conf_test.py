import pytest
import numpy as np

@pytest.fixture
def data_instance() -> np.array:
    """
    Fixture to create a sample data instance for testing.
    
    Returns
    -------
    np.array
        A numpy array of sample data points.
    """
    np.random.seed(111)
    return np.random.normal(loc=0, scale=1, size=1000)