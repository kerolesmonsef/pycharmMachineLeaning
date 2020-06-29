import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_wine as dataset
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class Own_PCA:
    def __init__(self, n_component=2):
        self.n_component = n_component
        super(Own_PCA, self).__init__()

    def fit(self, X, y=None):
        cov_mat = np.cov(np.transpose(X))
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        self.best = (eigen_pairs[i][1][:, np.newaxis] for i in range(self.n_component))
        self.w = np.hstack(self.best)
        # self.w = np.hstack((eigen_pairs[0][1][:, np.newaxis],eigen_pairs[1][1][:, np.newaxis]))
        return self

    def transform(self, X, y=None):
        new_X = np.dot(X, self.w)
        return new_X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X, y)


if __name__ == "__main__":
    X, y = dataset(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    steps = [
        ('scaler', StandardScaler()),
        ('pca_scaler', Own_PCA(n_component=2)),
        ('svc', SVC())
    ]

    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print('score : ', score)
    # pca = ()
    #
    # pca.fit(X_train)
    # X_pca = pca.transform(X)
