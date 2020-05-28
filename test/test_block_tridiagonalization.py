import unittest
import doctest
import nanonet.tb.block_tridiagonalization as btd
suite = doctest.DocTestSuite(btd)
unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    unittest.TextTestRunner(verbosity=2).run(suite)
