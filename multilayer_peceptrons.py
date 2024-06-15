import numpy as np
from perceptron import Perceptron


class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the Perceptron class above.
       Attributes:
          layers:  A python list with the number of elements per layer.
          bias:    The bias term. The same bias is used for all neurons.
          eta:     The learning rate."""

    def __init__(self, layers, bias=1.0, eta=0.5):
        """Return a new MLP object with the specified parameters."""
        self.layers = np.array(layers, dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = []  # The list of lists of neurons
        self.values = []  # The list of lists of output values
        self.d = []  # The list of lists of error terms (lowercase deltas)

        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:  # network[0] is the input layer, so it has no neurons
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i - 1], bias=self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)
        self.d = np.array([np.array(x) for x in self.d], dtype=object)

    def set_weights(self, w_init):
        """Set the weights.
           w_init is a 3D list with the weights for all but the input layer."""
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i + 1][j].set_weights(w_init[i][j])

    def print_weights(self):
        print()
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                print("Layer", i + 1, "Neuron", j, self.network[i][j].weights)
        print()

    def run(self, x):
        """Feed a sample x into the MultiLayer Perceptron."""
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i - 1])
        return self.values[-1]

    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorithm."""
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        # Challenge: Write the Backpropagation Algorithm.
        # Here you have it step by step:

        # STEP 1: Feed a sample to the network
        o = self.run(x)

        # STEP 2: Calculate the MSE
        error = y - o
        MSE = sum(error ** 2) / self.layers[-1]

        # STEP 3: Calculate the output error terms
        self.d[-1] = o * (1 - o) * error

        # STEP 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1, len(self.network) - 1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i + 1]):
                    fwd_error += self.network[i + 1][k].weights[h] * self.d[i + 1][k]
                self.d[i][h] = self.values[i][h] * (1 - self.values[i][h]) * fwd_error

        # STEPS 5 & 6: Calculate the deltas and update the weights
        for i in range(1, len(self.network)):
            for j in range(self.layers[i]):
                for k in range(self.layers[i - 1] + 1):
                    if k == self.layers[i - 1]:
                        delta = self.eta * self.d[i][j] * self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i - 1][k]
                    self.network[i][j].weights[k] += delta
        return MSE