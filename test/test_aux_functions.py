import unittest
from nanonet.tb.aux_functions import xyz2np


class Test(unittest.TestCase):

    def setUp(self):
        self.expected = ['Si1', 'Si2', 'Si3', 'Si4', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8']

    def test_labels_with_queal_indices(self):

        xyz_file = """12
            H cell
            Si2      0.0    0.0    0.0
            Si2      0.0    0.0    1.0
            Si3      0.0    0.0    2.0
            Si4      0.0    0.0    3.0
            H3       0.0    1.0    0.0
            H3       0.0   -1.0    0.0
            H3       0.0    1.0    1.0
            H3       0.0   -1.0    1.0
            H5       0.0    1.0    2.0
            H3       0.0   -1.0    2.0
            H7       0.0    1.0    3.0
            H3       0.0   -1.0    3.0
        """

        res, _ = xyz2np(xyz_file)
        self.assertListEqual(res, self.expected)

    def test_labels_with_indices(self):

        xyz_file = """12
            H cell
            Si1      0.0    0.0    0.0
            Si2      0.0    0.0    1.0
            Si3      0.0    0.0    2.0
            Si4      0.0    0.0    3.0
            H1       0.0    1.0    0.0
            H2       0.0   -1.0    0.0
            H3       0.0    1.0    1.0
            H4       0.0   -1.0    1.0
            H5       0.0    1.0    2.0
            H6       0.0   -1.0    2.0
            H7       0.0    1.0    3.0
            H8       0.0   -1.0    3.0
        """

        res, _ = xyz2np(xyz_file)
        self.assertListEqual(res, self.expected)

    def test_labels_without_indices(self):

        xyz_file = """12
            H cell
            Si      0.0    0.0    0.0
            Si      0.0    0.0    1.0
            Si      0.0    0.0    2.0
            Si      0.0    0.0    3.0
            H       0.0    1.0    0.0
            H       0.0   -1.0    0.0
            H       0.0    1.0    1.0
            H       0.0   -1.0    1.0
            H       0.0    1.0    2.0
            H       0.0   -1.0    2.0
            H       0.0    1.0    3.0
            H       0.0   -1.0    3.0
        """

        res, _ = xyz2np(xyz_file)
        self.assertListEqual(res, self.expected)


if __name__ == "__main__":
    unittest.main()
