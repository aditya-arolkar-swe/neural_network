import numpy as np


class ActivationFunction:
    def activation_fn(self, x: np.array) -> np.array:
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def activation_fn(self, x: np.array) -> np.array:
        return 1 / (1 + np.exp(-x))


class RectifiedLinearUnit(ActivationFunction):
    def activation_fn(self, x: np.array) -> np.array:
        return np.maximum(x, np.zeros(shape=x.shape))
