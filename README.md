import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/content/sample_data/Advertising Budget and Sales.csv")
df.head()

df.isnull().sum() * 100 / df.shape[0]
df.isnull().sum()

sns.set(style="whitegrid")
sns.lmplot(x='TV Ad Budget ($)', y='Sales ($)', data=df, aspect=1.5, height=6)
plt.show()

X = df['TV Ad Budget ($)']
y = df['Sales ($)']

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3,
random_state=100)

X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(y_train, X_train_sm).fit()

print(lr.params)
print(lr.summary())

X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
