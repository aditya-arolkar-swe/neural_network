from typing import List
import numpy as np
from activation_functions import ActivationFunctions


class Perceptron:
    """
    A single neuron
    Attributes:
        inputs: The number of inputs in the perceptron, not counting the bias.
        bias:   The bias term, default 1.0.
    """

    def __init__(self, inputs: List[float], activation_fn_obj: ActivationFunctions, bias: int = 1.0):
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias
        self.activation_fn = activation_fn_obj.activation_fn

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        inputs = np.append(x, self.bias)  # all of x's input plus the trivial bias term of 1
        x_sum = np.dot(inputs, self.weights)  # i_0 * w_0 + i_1 * w_1 + ... + i_n * w_n
        output = self.activation_fn(x_sum)  # non-linear activation function to stabilize output

        return output

    def set_weights(self, w_init):
        """Set the weights. w_init is a python list with the weights."""
        self.weights = np.array(w_init)


