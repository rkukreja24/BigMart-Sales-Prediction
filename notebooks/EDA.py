# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Label Encoding simple
from sklearn.preprocessing import LabelEncoder
# Now quick model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Load datasets
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Display shape
print("Train shape:", train.shape)
print("Test shape:", test.shape)

# Preview data
train.head()

# Missing values
train.isnull().sum()
test.isnull().sum()

sns.histplot(train['Item_Outlet_Sales'], kde=True)
plt.title('Sales Distribution')
plt.show()

sns.boxplot(data=train, x='Outlet_Type', y='Item_Outlet_Sales')

# We will analyze only the training set
train['Item_Identifier'].value_counts(normalize = True)
train['Item_Identifier'].value_counts().plot.hist()
plt.title('Variants of items available')
plt.legend()
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='Item_Fat_Content', data=train)
plt.xticks(rotation=45)

# checking the different items in Item Type

train['Item_Type'].value_counts()

train['Item_Type'].value_counts(normalize = True)
train['Item_Type'].value_counts().plot.bar()
plt.title('Different types of item available in the store')
plt.show()

# checking different types of item in Outlet Type

train['Outlet_Type'].value_counts()
train['Outlet_Type'].value_counts(normalize = True)
train['Outlet_Type'].value_counts().plot.bar()
plt.title('Different types of outlet types in the store')
plt.show()

for col in train.select_dtypes(include='object'):
    print(col, ":", train[col].unique())

# combining the train and test dataset
data = pd.concat([train, test])
print(data.shape)

# checking unique values in the columns of train dataset
data.apply(lambda x: len(x.unique()))

data.isnull().sum()

# imputing missing values

data['Item_Weight'] = data['Item_Weight'].replace(0, np.NaN)
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace = True)

data['Outlet_Size'].fillna(data['Outlet_Size'].mode()[0], inplace = True)

data['Item_Outlet_Sales'] = data['Item_Outlet_Sales'].replace(0, np.NaN)
data['Item_Outlet_Sales'].fillna(data['Item_Outlet_Sales'].mode()[0], inplace = True)

data.isnull().sum()

# combining reg, Regular and Low Fat, low fat and, LF

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
data['Item_Fat_Content'].value_counts()

# determining the operation peroid of a time

data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].value_counts()

data.apply(LabelEncoder().fit_transform)

# Save identifiers
item_ids = test['Item_Identifier']
outlet_ids = test['Outlet_Identifier']

# one hot encoding
data = pd.get_dummies(data)
print(data.shape)

# splitting the data into dependent and independent variables
x = data.drop('Item_Outlet_Sales', axis = 1)
y = data.Item_Outlet_Sales

print(x.shape)
print(y.shape)

# splitting the dataset into train and test
train = data.iloc[:8523,:]
test = data.iloc[8523:,:]

print(train.shape)
print(test.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Linear Regression
LR_model = LinearRegression()
LR_model.fit(x_train, y_train)

# predicting the  test set results
y_pred = LR_model.predict(x_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print('RMSE :', np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# AdaBoost Regressor
abr_model= AdaBoostRegressor(n_estimators = 100)
abr_model.fit(x_train, y_train)

# predicting the test set results
y_pred = abr_model.predict(x_test)

# RMSE
mse = mean_squared_error(y_test, y_pred)
print("RMSE :", np.sqrt(mse))

# XgBoost Regressor
xbg_model = GradientBoostingRegressor()
xbg_model.fit(x_train, y_train)

# predicting the test set results
y_pred = xbg_model.predict(x_test)
print(y_pred)

# Calculating the root mean squared error
print("RMSE :", np.sqrt(((y_test - y_pred)**2).sum()/len(y_test)))

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators = 100 , n_jobs = -1)
rf_model.fit(x_train, y_train)

# predicting the  test set results
y_pred = rf_model.predict(x_test)
print(y_pred)

# finding the mean squared error and variance
mse = mean_squared_error(y_test, y_pred)
print("RMSE :",np.sqrt(mse))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

print("Result :",rf_model.score(x_train, y_train))

# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(x_train, y_train)

# predicting the test set results
y_pred = dt_model.predict(x_test)
print(y_pred)

print(" RMSE : " , np.sqrt(((y_test - y_pred)**2).sum()/len(y_test)))

# Support vector machine
svr_model = SVR()
svr_model.fit(x_train, y_train)

# predicting the x test results
y_pred = svr_model.predict(x_test)

# Calculating the RMSE Score
mse = mean_squared_error(y_test, y_pred)
print("RMSE :", np.sqrt(mse))

# Predict on actual test data (from row 8523 onward)
final_predictions = xbg_model.predict(x[8523:])

# Prepare the submission dataframe
submission = pd.DataFrame({
    'Item_Identifier': item_ids,
    'Outlet_Identifier': outlet_ids,
    'Item_Outlet_Sales': final_predictions
})

# Save to CSV
submission.to_csv('../submission/submission.csv', index=False)