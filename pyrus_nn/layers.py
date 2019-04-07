# -*- coding: utf-8 -*-


class Layer:
    pass


class Dense(Layer):

    def __init__(self, n_input: int, n_output: int, activation: str):
        """
        A fully connected layer interface

        Parameters
        ----------
        n_input: int
            Number of input neurons to the layer
        n_output: int
            Number of output neurons from the layer
        activation: str
            Layer activation function, one of (sigmoid, tanh, softmax, linear)
        """
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
