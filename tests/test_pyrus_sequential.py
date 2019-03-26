# -*- coding: utf-8 -*-

import unittest

from pyrus_nn.models import Sequential
from pyrus_nn.rust.pyrus_nn import PyrusSequential


class SequentialTestCase(unittest.TestCase):

    def test_rust_raw_init(self):
        """Basic init test"""
        _nn = PyrusSequential(lr=0.05, n_epoch=2)

    def test_py_interface_init(self):
        """
        Basic init test for py wrapper
        """
        _model = Sequential(lr=0.01, n_epochs=2)

    def test_py_interface_add_dense(self):
        model = Sequential(lr=0.01, n_epochs=2)
        model.add_dense(4, 5)
