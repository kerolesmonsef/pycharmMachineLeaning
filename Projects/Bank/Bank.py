import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Churn-Modelling.csv')
y = data['Exited']
data = data.drop(['CustomerId', 'Surname', 'RowNumber'], axis=1)

X = data.iloc[:, 0:10]
y = data.iloc[:, 10]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

X_upsampled, y_upsampled = resample(X_train[y_train == 1],
                                    y_train[y_train == 1],
                                    replace=True,
                                    n_samples=X_train[y_train == 1].shape[0],
                                    random_state=123)

# X_train = np.vstack((X_train[y_train == 0], X_upsampled))
# y_train = np.hstack((y_train[y_train == 0], y_upsampled))

# dfff = X_train , y_train
# sm = SMOTE(random_state=27)
# X_train, y_train = sm.fit_sample(X_train, y_train)

print('begin')

logistic_regression = LogisticRegression(random_state=1, n_jobs=-1)

decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

k_neighbors = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski', n_jobs=-1)

logistic_regression_pipe = Pipeline([['sc', StandardScaler()],
                                     ['clf', logistic_regression]])
KNeighbors_pipe = Pipeline([['sc', StandardScaler()],
                            ['clf', k_neighbors]])

ada_boost_classifier_pipe = Pipeline([
    ['clf', AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=2),
        n_estimators=500,
        learning_rate=0.1,
        random_state=1)]
])
gaussian_nb = Pipeline([('sc', StandardScaler()),
                        ('clf', GaussianNB())])

svm_pipe = Pipeline([('sc', StandardScaler()),
                     ('clf', SVC(kernel='poly'))])

random_forest = Pipeline([
    ['clf', RandomForestClassifier(criterion='entropy', n_estimators=100, n_jobs=-1)]])

py_voting_classifier = VotingClassifier(estimators=[
    ('lr', logistic_regression_pipe), ('dt', decision_tree), ('KNN', KNeighbors_pipe), ('rf', random_forest),
    ('svm', svm_pipe)],
    voting='hard', n_jobs=-1)

clf_labels = ['Logistic regression', 'Decision tree', 'KNN', 'gaussian_nb', 'svm', 'AdaBoostClassifier',
              'Random Forest', 'Majority voting', 'py_voting_classifier']

all_clf = [logistic_regression_pipe, decision_tree, KNeighbors_pipe, gaussian_nb, svm_pipe, ada_boost_classifier_pipe,
           random_forest, py_voting_classifier]

tabel = PrettyTable()
tabel.field_names = ["estimator", "score", 'f1 score', 'recall score', 'precision score']
for clf, label in zip(all_clf, clf_labels):
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_test, y_pred)
    tabel.add_row(
        [label, score, f1_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)])
    print(label, ' : ', score)

print(tabel)
input('press any key to continue ....')
