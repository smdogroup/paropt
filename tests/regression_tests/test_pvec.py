from paropt import ParOpt
import unittest
import numpy as np


class Prob(ParOpt.Problem):
    """
    A helper problem instance
    """

    def __init__(self, comm, nvars, ncon):
        super().__init__(comm, nvars, ncon)


class PVecTest(unittest.TestCase):
    N_PROCS = 2  # num of procs used

    def setUp(self):
        # Get rank and size
        self.rank = self.comm.rank
        self.size = self.comm.size

        # Create problem
        self.nvars = self.rank + 10
        ncon = 1
        self.prob = Prob(self.comm, self.nvars, ncon)

        return

    def test_setitem_list_list(self):
        # Create vector
        vec = self.prob.createDesignVec()

        # Populate vector
        indices = [i for i in range(self.nvars)]
        vals = np.random.rand(self.nvars)
        vec[indices] = vals

        # Test
        for i in indices:
            self.assertEqual(vec[i], vals[i])

    def test_setitem_list_single(self):
        # Create vector
        vec = self.prob.createDesignVec()

        # Populate vector
        indices = [i for i in range(self.nvars)]
        vec[indices] = 1.23

        # Test
        for i in indices:
            self.assertEqual(vec[i], 1.23)

    def test_getitem_list(self):
        # Create vector
        vec = self.prob.createDesignVec()
        vec[:] = 1.23

        # Populate vector
        indices = [i for i in range(self.nvars)]

        # Test __getitem__
        for i in range(self.nvars):
            self.assertEqual(vec[i], 1.23)
