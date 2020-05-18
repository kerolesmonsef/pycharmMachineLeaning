import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data_banknote_authentication.csv')
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
trainScore = clf.score(X_train,y_train)
testScore = clf.score(X_test,y_test)

print('trainScore : ',trainScore)
print('testScore : ',testScore)