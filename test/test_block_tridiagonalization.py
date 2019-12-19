import unittest
import doctest
import tb.block_tridiagonalization as btd
suite = doctest.DocTestSuite(btd)


if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)