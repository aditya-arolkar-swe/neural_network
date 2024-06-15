import numpy as np


class ActivationFunctions:
    def activation_fn(self, x: np.array) -> float:
        raise NotImplementedError


class Sigmoid(ActivationFunctions):
    def activation_fn(self, x: np.array) -> float:
        return 1 / (1 + np.exp(-x))