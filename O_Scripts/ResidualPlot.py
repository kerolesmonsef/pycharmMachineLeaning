import numpy as np
import matplotlib.pyplot as plt


class ResidualPlot:

    def __init__(self, y, pred_y):
        self.__y = np.array(y)
        self.__predicted_y = np.array(pred_y)
        self.__residual = self.__y - self.__predicted_y

    def Y_residual(self):
        plt.figure(np.random.randint(2000))
        plt.scatter(self.__y, self.__residual)
        plt.axhline(0, color='red')
        plt.xlabel("Y")
        plt.ylabel("Residua")
        plt.show()

    def Y_hat_residual(self):
        plt.figure(np.random.randint(2000))
        plt.scatter(self.__predicted_y, self.__residual)
        plt.axhline(0, color='red')
        plt.xlabel("Y_hat")
        plt.ylabel("Residua")
        plt.show()
