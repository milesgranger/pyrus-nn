# -*- coding: utf-8 -*-

from pyrus_nn.rust.pyrus_nn import PyrusSequential


class Sequential:

    _model: PyrusSequential

    def __init__(self, lr: float, n_epochs: int):
        self._model = PyrusSequential(lr, n_epochs)

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def add_dense(self, n_input: int, n_output: int):
        self._model.add_dense(n_input, n_output)
