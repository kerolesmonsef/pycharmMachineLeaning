import pandas as pd
import numpy as np
import math, datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
style.use('ggplot')

df = pd.read_csv("Data/wik2000-2018.csv")
df.index = df['Date']
df = df[['Date', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Date', 'Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)  # the DF contain Nan In the End
X = np.array(df.drop(['label', 'Date'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # get the Xs that contains (NaN Ys)
X = X[:-forecast_out]  # The X does Not Contain NaN ys

df.dropna(inplace=True)
y = np.array(df['label'])
# X = preprocessing.scale(X)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

clf = LinearRegression(n_jobs=-1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print("Accuracy = ", accuracy)
forecast_set = clf.predict(X_lately)  # This is what we need the prediction to the future

###################################
## This for Plotting The Result ##
##################################
df['Forecast'] = np.nan
last_date = df.iloc[-1]['Date']
last_unix = datetime.datetime.strptime(last_date, '%Y-%m-%d').timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_day = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_day] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df[forecast_col].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel("Date")
plt.ylabel('Price')
plt.show()
