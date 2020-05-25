import numpy as np, random, os
import matplotlib.pyplot as plt


"""
    this Neural Network for Only { And | Or | Nor | Nand } Operation
"""


class GatesNeuralNetwork:
    def __init__(self, data, iter_count, operation='OR'):
        self.data = np.array(data)
        self.iter_count = iter_count
        self.lr = 1  # learning rate
        self.bias = 1  # value of bias
        self.weights = [0, 0, 0]
        self.operation = operation

    def Perceptron(self, input1, input2, output):
        outputP = input1 * self.weights[0] + input2 * self.weights[1] + self.bias * self.weights[2]
        if outputP > 0:  # activation function (here Heaviside)
            outputP = 1
        else:
            outputP = 0
        error = output - outputP
        self.weights[0] += error * input1 * self.lr
        self.weights[1] += error * input2 * self.lr
        self.weights[2] += error * self.bias * self.lr

    def train(self):
        for _ in range(self.iter_count):
            for row in self.data:
                self.Perceptron(row[0], row[1], row[2]);

    def predict(self, x, y):
        outputP = x * self.weights[0] + y * self.weights[1] + self.bias * self.weights[2]
        if outputP > 0:  # activation function
            outputP = 1
        else:
            outputP = 0
        print(x, self.operation, y, "is : ", outputP)
        return outputP

    def score(self):
        for row in self.data:
            self.predict(row[0], row[1])

    def draw(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.data[:, -1])
        x = np.linspace(0, 1)
        y = -(self.weights[0] * x + self.weights[2]) / self.weights[1]
        plt.plot(x, y)
        # x1 = 0
        # y1 = self.weights[1]
        plt.show()


if __name__ == "__main__":
    data = [[1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1]]
    clf = GatesNeuralNetwork(data, 50, "AND")
    clf.train()
    clf.score()
    # clf.draw()
