import numpy as np
import matplotlib.pyplot as plt


class ResidualPlot:

    def __init__(self, y, pred_y):
        self.__y = np.array(y)
        self.__predicted_y = np.array(pred_y)
        self.__residual = self.__predicted_y - self.__y
        self.__residual2 = self.__y - self.__predicted_y

    def draw(self):
        plt.figure(np.random.randint(2000))
        plt.scatter(self.__predicted_y, self.__predicted_y - self.__y)
        plt.axhline(0, color='red')
        plt.xlabel("Y Hat")
        plt.ylabel("Residua")
        plt.show()

    def draw2(self):
        plt.figure(np.random.randint(2000))
        plt.scatter(self.__y, self.__y - self.__predicted_y)
        plt.axhline(0, color='red')
        plt.xlabel("Y")
        plt.ylabel("Residua")
        plt.show()
