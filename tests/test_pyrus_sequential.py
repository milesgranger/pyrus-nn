# -*- coding: utf-8 -*-

import unittest


class SequentialTestCase(unittest.TestCase):

    def test_rust_raw_init(self):
        """Basic init test"""
        from pyrus_nn.rust.pyrus_nn import PyrusSequential
        nn = PyrusSequential(lr=0.05)

    def test_py_interface_init(self):
        """
        Basic init test for py wrapper
        """
        from pyrus_nn.models import Sequential
        model = Sequential(lr=0.01, n_epochs=2)
