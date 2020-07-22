import unittest
import numpy as np
from nanonet.tb.aux_functions import compute_edge, blocksandborders_constrained


class Test(unittest.TestCase):
    """ """

    def setUp(self):
        """ """
        a = np.array([[1, 1], [1, 1]])
        b = np.zeros((2, 2))
        self.test_matrix = np.block([[a, b, b], [b, a, b], [b, b, a]])
        self.test_matrix[0, 2] = 2
        self.test_matrix[2, 0] = 2
        self.test_matrix[2, 4] = 3
        self.test_matrix[4, 2] = 3

        self.edge = np.array([3, 3, 5, 5, 6, 6])
        self.edge_star = np.array([2, 4, 4, 6, 6, 6])

    def test_compute_edge(self):
        """ """

        edge, edge1 = compute_edge(self.test_matrix)
        np.testing.assert_allclose(edge, self.edge, atol=1e-4)
        np.testing.assert_allclose(edge1, self.edge_star, atol=1e-4)

    def test_blocksandborders_constrained_0_0(self):
        """ """
        ans = blocksandborders_constrained(0, 0, self.edge, self.edge_star)
        self.assertListEqual(ans, [1, 3, 1, 1])

    def test_blocksandborders_constrained_1_1(self):
        """ """
        ans = blocksandborders_constrained(1, 1, self.edge, self.edge_star)
        self.assertListEqual(ans, [1, 3, 1, 1])

    def test_blocksandborders_constrained_2_2(self):
        """ """
        ans = blocksandborders_constrained(2, 2, self.edge, self.edge_star)
        self.assertListEqual(ans, [2, 2, 2])

    def test_blocksandborders_constrained_3_3(self):
        """ """
        ans = blocksandborders_constrained(3, 3, self.edge, self.edge_star)
        self.assertListEqual(ans, [3, 3])

    def test_blocksandborders_constrained_4_4(self):
        """ """
        ans = blocksandborders_constrained(4, 4, self.edge, self.edge_star)
        self.assertListEqual(ans, [6])

    def test_blocksandborders_constrained_5_5(self):
        """ """
        ans = blocksandborders_constrained(6, 6, self.edge, self.edge_star)
        self.assertListEqual(ans, [6])

    def test_blocksandborders_constrained_2_4(self):
        """ """
        ans = blocksandborders_constrained(2, 4, self.edge, self.edge_star)
        self.assertListEqual(ans, [2, 4])

    def test_blocksandborders_constrained_2_3(self):
        """ """
        ans = blocksandborders_constrained(2, 3, self.edge, self.edge_star)
        self.assertListEqual(ans, [2, 1, 3])

    def test_blocksandborders_constrained_3_2(self):
        """ """
        ans = blocksandborders_constrained(3, 2, self.edge, self.edge_star)
        self.assertListEqual(ans, [4, 2])


if __name__ == "__main__":
    unittest.main()
