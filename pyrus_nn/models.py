# -*- coding: utf-8 -*-

from pyrus_nn.rust.pyrus_nn import PyrusSequential
from pyrus_nn import layers


class Sequential:

    # This is the actual Rust implementation with Python interface
    _model: PyrusSequential

    def __init__(self, lr: float, n_epochs: int):
        self._model = PyrusSequential(lr, n_epochs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def add(self, layer: layers.Layer):
        if isinstance(layer, layers.Dense):
            self._model.add_dense(layer.n_input, layer.n_output)
