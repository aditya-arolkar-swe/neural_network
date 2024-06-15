from multilayer_peceptrons import MultiLayerPerceptron


if __name__ == '__main__':
    mlp = MultiLayerPerceptron(layers=[2, 2, 1])
    print("\nTraining Neural Network as an XOR Gate...\n")
    for i in range(3000):
        mse = 0.0
        mse += mlp.bp([0, 0], [0])
        mse += mlp.bp([0, 1], [1])
        mse += mlp.bp([1, 0], [1])
        mse += mlp.bp([1, 1], [0])
        mse = mse / 4
        if i % 100 == 0:
            print(mse)

    mlp.print_weights()

    print("MLP:")
    print("0 0 = {0:.10f}".format(mlp.run([0, 0])[0]))
    print("0 1 = {0:.10f}".format(mlp.run([0, 1])[0]))
    print("1 0 = {0:.10f}".format(mlp.run([1, 0])[0]))
    print("1 1 = {0:.10f}".format(mlp.run([1, 1])[0]))
