import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ABCtransaformer(TransformerMixin, BaseEstimator):
    abc = {
        "a": 0,
        "b": 0,
        "c": 0,
        "d": 0,
        "e": 0,
        "f": 0,
        "g": 0,
        "h": 0,
        "i": 0,
        "j": 0,
        "k": 0,
        "l": 0,
        "m": 0,
        "n": 0,
        "o": 0,
        "p": 0,
        "q": 0,
        "r": 0,
        "s": 0,
        "t": 0,
        "u": 0,
        "v": 0,
        "w": 0,
        "x": 0,
        "y": 0,
        "z": 0,
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "6": 0,
        "7": 0,
        "8": 0,
        "9": 0,
    }

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self

    def __transform(self, S) -> np.array([]):
        abcDict = self.abc.copy()

        for char_i in S:
            if char_i in abcDict:
                abcDict[char_i] += 1
        return np.fromiter(abcDict.values(), dtype=float)

    def transform(self, X, copy=None):
        if type(X) == str:
            return self.__transform(X)

        trans = []

        for doc in X:
            trans.append(self.__transform(doc))
        return np.array(trans)
