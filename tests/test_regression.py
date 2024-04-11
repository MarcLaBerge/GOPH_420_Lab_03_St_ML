import numpy as np
import unittest
from lab_03.regression import multi_regress

"""
Testing the regression method against quiz 3, using excel to compare/set expected
"""

class TestRegressionQuad(unittest.TestCase):

    
    def setUp(self):
        #Testing the algorithm with quadratic example
        #Y part of the quiz array
        self.y = np.array([1.34, 2.14, 3.0, 3.69, 4.73, 4.47, 5.38, 5.38])

        #Z part of the quiz array
        self.Z = np.array([[1, 2, 4],
                           [1, 4, 16],
                           [1, 10, 100],
                           [1, 20, 400],
                           [1, 30, 900],
                           [1, 60, 3600],
                           [1, 120, 14400],
                           [1, 180, 32400]])

    def test_quad(self):
        #expected values are from quiz 3
        a_expect = np.array([2.1389884536, 0.0592360745, -0.0002357483])
        e_expect = np.array([-0.9165176096, -0.2321607795, 0.2922256280, 0.4605893638, 1.0261027537, -0.3744591519, -0.4725423017, 0.2167620971])
        rsq_expect = 0.8315119699
        a, e, rsq = multi_regress(self.y, self.Z)
        #Compare that they are close enough
        self.assertTrue(np.allclose(a_expect, a, atol = 1e-9))
        self.assertTrue(np.allclose(e_expect, e, atol = 1e-9))
        self.assertAlmostEqual(rsq_expect, rsq)


class TestRegressionLinear(unittest.TestCase):
    #Testing the algorithm with linear example
    def setUp(self):
        self.y = np.array([1/1.34, 1/2.14, 1/3.0, 1/3.69, 1/4.73, 1/4.47, 1/5.38, 1/5.38])
        self.Z = np.array([[1, 1/2],
                           [1, 1/4],
                           [1, 1/10],
                           [1, 1/20],
                           [1, 1/30],
                           [1, 1/60],
                           [1, 1/120],
                           [1, 1/180]])

    def test_lin(self):
        #again, expected values are from quiz 3
        a_expect = np.array([0.1940956276, 1.1121683843])
        e_expect = np.array([-0.0039111630, -0.0048480040, 0.0280208674, 0.0212986633, -0.0197514165, 0.0110818792, -0.0174900915, -0.0144007349])
        rsq_expect = 0.99122339
        a, e, rsq = multi_regress(self.y, self.Z)
        #Compare that they are close enough
        self.assertTrue(np.allclose(a_expect, a, atol = 1e-9))
        self.assertTrue(np.allclose(e_expect, e, atol = 1e-9))
        self.assertAlmostEqual(rsq_expect, rsq)