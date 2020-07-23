from O_Scripts.ABCtransformer import ABCtransaformer
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
import pyprind

data = pd.read_csv('E:\study\program_language\AI\datasests\imdb_50000review\movie_data.csv')
data = data[:25000]
abc_trans = ABCtransaformer()
y = data['sentiment']
X = data['review']
X = abc_trans.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)

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

pbar = pyprind.ProgBar(len(all_clf))
tabel = PrettyTable()
tabel.field_names = ["estimator", "score", 'f1 score', 'recall score', 'precision score']
for clf, label in zip(all_clf, clf_labels):
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = accuracy_score(y_test, y_pred)
    tabel.add_row(
        [label, score, f1_score(y_test, y_pred), recall_score(y_test, y_pred), precision_score(y_test, y_pred)])
    print(label, ' : ', score)
    pbar.update()

print(tabel)
input('press any key to continue ....')
