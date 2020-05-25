from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sns.set()
np.set_printoptions(precision=4)


class test:
    _w_matrix = []

    def LDA(self, X, y):
        df = X.join(pd.Series(y, name='class'))
        _, dim = X.shape
        class_feature_means = {}
        for c, rows in df.groupby('class'):
            class_feature_means[c] = rows.mean()

        within_class_scatter_matrix = np.zeros((dim, dim))
        for c, rows in df.groupby('class'):
            rows = rows.drop(['class'], axis=1)
            s = np.zeros((dim, dim))
            for index, row in rows.iterrows():
                x, mc = row.values.reshape(dim, 1), class_feature_means[c].values.reshape(dim, 1)
                s += (x - mc).dot((x - mc).T)
            within_class_scatter_matrix += s

        feature_means = df.mean()
        between_class_scatter_matrix = np.zeros((dim, dim))
        for c in class_feature_means:
            n = len(df.loc[df['class'] == c].index)
            mc, m = class_feature_means[c].values.reshape(dim, 1), feature_means.values.reshape(dim, 1)
            between_class_scatter_matrix += n * (mc - m).dot((mc - m).T)

        matc = np.linalg.inv(within_class_scatter_matrix).dot(between_class_scatter_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(matc)

        pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(len(eigen_values))]
        pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

        eigen_value_sums = sum(eigen_values)
        self._w_matrix = np.hstack(
            (pairs[0][1].reshape(dim, 1), pairs[1][1].reshape(dim, 1), pairs[2][1].reshape(dim, 1))).real
        X = np.array(X.dot(self._w_matrix))

        le = LabelEncoder()
        y = le.fit_transform(df['class'])
        return X, y


if __name__ == "__main__":
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.Categorical.from_codes(wine.target, wine.target_names)
    lda = test()
    X_lda, y = lda.LDA(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('X before', X_train.shape)
    print('accuracy before LDA', accuracy)

    print("------------------ after --------")
    X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.3, random_state=42)
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('X after', X_train.shape)
    print('accuracy after LDA', accuracy)
