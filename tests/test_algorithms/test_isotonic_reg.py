import numpy as np
from bezierv.algorithms.isotonic_reg import bounded_iso_mean, project

def test_bounded_iso_mean():
    vals = np.array([-1,0.1,-0.5,0.4,0.7,1, 0.33])
    iso = bounded_iso_mean(vals.copy(), w=np.ones(len(vals)), a=np.zeros(len(vals)), b=np.ones(len(vals)))
    expected = np.array([0.0, 0.0, 0.0, 0.4, 0.6766667, 0.6766667, 0.6766667])
    np.testing.assert_allclose(iso, expected)

    vals = np.array([-1, -4, 3, 9.8, 7.6, -6.99, 0.343])
    iso = bounded_iso_mean(vals.copy(), w=np.ones(len(vals)), a=-5*np.ones(len(vals)), b=8*np.ones(len(vals)))
    expected = np.array([-2.5000, -2.5000,  2.7506,  2.7506,  2.7506,  2.7506,  2.7506])
    np.testing.assert_allclose(iso, expected)

def test_project():
    vals = np.array([-10,3,2,7,1])
    projected = project(vals, lower=-7, upper=8)
    expected = np.array([-7,2.5,2.5,7,8])
    np.testing.assert_allclose(projected, expected)