from typing import List, Union
import numpy as np
from activation_functions import ActivationFunction


class Perceptron:
    """
    A single neuron
    Attributes:
        inputs: The number of inputs in the perceptron, not counting the bias.
        bias:   The bias term, default 1.0.
    """

    def __init__(self, inputs: Union[np.array, List[float]], activation_fn_obj: ActivationFunction, bias: float = 1.0) -> None:
        self.weights = (np.random.rand(inputs + 1) * 2) - 1
        self.bias = bias
        self.activation_fn = activation_fn_obj.activation_fn

    def run(self, inputs: List[float]) -> np.array:
        """ Run the perceptron """
        # all of x's input plus the trivial bias term of 1
        inputs_plus_bias = np.append(inputs, self.bias)

        # for length n input, gives us i_0 * w_0 + i_1 * w_1 + ... + i_n * w_n + bias * w_n+1
        x_sum = np.dot(inputs_plus_bias, self.weights)

        # non-linear activation function to stabilize output and allow complex classification
        output = self.activation_fn(x_sum)

        return output

    def set_weights(self, w_init: List[float]) -> None:
        """ Set the weights to w_init """
        self.weights = np.array(w_init)


