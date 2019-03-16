# -*- coding: utf-8 -*-

import unittest
from pyrus_nn.rust.pyrus_nn import PyrusSequential


class SequentialTestCase(unittest.TestCase):

    def test_init(self):
        """Basic init test"""
        nn = PyrusSequential(lr=0.05)
