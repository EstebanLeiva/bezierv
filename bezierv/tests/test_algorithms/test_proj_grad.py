from bezierv.algorithms import proj_grad2
from bezierv.classes import Bezierv, DistFit

def test_projgrad(data_instance):
    """
    Test the proj_grad2 algorithm with a sample data instance.
    
    This test checks if the proj_grad2 algorithm can fit a Bezierv instance to the provided data.
    """

    distfit = DistFit(data=data_instance)
    bezierv = distfit.fit(method='projgrad')
    


