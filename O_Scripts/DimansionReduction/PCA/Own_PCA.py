import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_wine as dataset
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC


class Own_PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        cov_mat = np.cov(np.transpose(X))
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        self.best_pairs = (eigen_pairs[i][1][:, np.newaxis] for i in range(self.n_components))
        self.w = np.hstack(self.best_pairs)
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
        ('pca_scaler', Own_PCA(n_components=2)),
        ('svc', SVC())
    ]

    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    score = pipe.score(X_test, y_test)
    print('score : ', accuracy_score(y_pred, y_test))
    print('f1_score : ', f1_score(y_test, y_pred, average='macro'))
    print('recall_score : ', recall_score(y_pred, y_test, average='macro'))
    # pca = ()
    #
    # pca.fit(X_train)
    # X_pca = pca.transform(X)
