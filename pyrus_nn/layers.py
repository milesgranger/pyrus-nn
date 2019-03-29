# -*- coding: utf-8 -*-


class Layer:
    pass


class Dense(Layer):

    def __init__(self, n_input: int, n_output: int):
        self.n_input = n_input
        self.n_output = n_output
