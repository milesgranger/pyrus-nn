# -*- coding: utf-8 -*-

from typing import Iterable

from pyrus_nn.rust.pyrus_nn import PyrusSequential
from pyrus_nn import layers


class Sequential:

    # This is the actual Rust implementation with Python interface
    _model: PyrusSequential

    def __init__(self, lr: float, n_epochs: int):
        """
        Initialize the model.

        Parameters
        ----------
        lr: float
            The learning rate of the model
        n_epochs: int
            How many epochs shall it do for training
        """
        self._model = PyrusSequential(lr, n_epochs)

    def fit(self, X: Iterable[float], y: Iterable[float]):
        """
        Fit the model using X and y. Each of which would be a 2d iterable.

        For example::

            X = [[1, 2, 3], [4, 5, 6]]
            y = [[1], [2]]

        Parameters
        ----------
        X: Iterable
        y: Iterable

        Returns
        -------
        self
        """
        self._model.fit(X, y)
        return self

    def add(self, layer: layers.Layer):
        """
        Add a layer to this network

        Parameters
        ----------
        layer: pyrus_nn.layers.Layer
            A layer compatible with the previous layer

        Returns
        -------
        None
        """
        if isinstance(layer, layers.Dense):
            self._model.add_dense(layer.n_input, layer.n_output)
