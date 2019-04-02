# -*- coding: utf-8 -*-

import pytest
import numpy as np

from pyrus_nn.models import Sequential
from pyrus_nn import layers
from pyrus_nn.rust.pyrus_nn import PyrusSequential


def test_rust_raw_init():
    """Basic init test"""
    _nn = PyrusSequential(lr=0.05, n_epoch=2)


def test_py_interface_init():
    """
    Basic init test for py wrapper
    """
    _model = Sequential(lr=0.01, n_epochs=2)


@pytest.mark.parametrize("layer", [
    (layers.Dense(2, 3),),
    (layers.Dense(2, 3),),
    (layers.Dense(128, 256),)
])
def test_py_interface_add_layer(layer):
    model = Sequential(lr=0.01, n_epochs=2)
    model.add(layer)


def test_fit_predict():
    model = Sequential(lr=0.01, n_epochs=2)
    model.add(layers.Dense(2, 4))
    model.add(layers.Dense(4, 1))

    X = np.random.random(size=100).reshape(-1, 2)
    y = np.random.randint(0, 10, size=50).reshape(-1, 1)
    model.fit(X, y)
