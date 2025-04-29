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
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

for col in train.select_dtypes(include='object'):
    print(col, ":", train[col].unique())

train['Outlet_Age'] = 2025 - train['Outlet_Establishment_Year']

# Quick simple filling
train['Item_Weight'].fillna(train['Item_Weight'].mean(), inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].mean(), inplace=True)

train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True)
test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0], inplace=True)

le = LabelEncoder()
for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

X = train.drop(['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)
y = train['Item_Outlet_Sales']

# Split into train and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on validation set
rf_predictions = model.predict(X_val)

# Calculate RMSE
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_predictions))
print(f'Random Forest RMSE: {rf_rmse}')

# 1. Fix Fat Content
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({0:'LF',1:'Regular',2:'reg', 3:'Low Fat', 4:'low fat'})
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({0:'LF',1:'Regular',2:'reg', 3:'Low Fat', 4:'low fat'})

# 2. Replace zero visibility with mean visibility per item
train.loc[train['Item_Visibility'] == 0, 'Item_Visibility'] = train['Item_Visibility'].mean()
test.loc[test['Item_Visibility'] == 0, 'Item_Visibility'] = test['Item_Visibility'].mean()

# 3. Create Outlet Age
train['Outlet_Age'] = 2025 - train['Outlet_Establishment_Year']
test['Outlet_Age'] = 2025 - test['Outlet_Establishment_Year']

# 4. Create Item Visibility Mean Ratio
visibility_avg = train.groupby('Item_Identifier')['Item_Visibility'].mean()
train['Item_Visibility_MeanRatio'] = train.apply(lambda x: x['Item_Visibility']/visibility_avg[x['Item_Identifier']], axis=1)
test['Item_Visibility_MeanRatio'] = test.apply(lambda x: x['Item_Visibility']/visibility_avg.get(x['Item_Identifier'], 1), axis=1)

# 5. Create Broad Item Categories
def item_category(x):
    if x in [4, 15, 6, 10, 1, 5, 7, 8, 9, 11, 13, 16]: # Food-related encoded Item_Type
        return 'Food'
    elif x in [12, 14, 17]:
        return 'Non-Consumable'
    else:
        return 'Drinks'

train['Item_Category'] = train['Item_Type'].apply(item_category)
test['Item_Category'] = test['Item_Type'].apply(item_category)

# 6. Label Encode new features
for col in ['Item_Fat_Content', 'Item_Category']:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 7. Final Feature Set
drop_cols = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year', 'Item_Type']
X_train = train.drop(columns=drop_cols + ['Item_Outlet_Sales'])
y_train = train['Item_Outlet_Sales']
X_test = test.drop(columns=drop_cols)

# Split the train data into 80% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = le.transform(X_val[col])

# Train XGBoost
xgb = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)

# Predict on the validation set
val_predictions = xgb.predict(X_val)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"RMSE: {rmse}")


# Predict on the test set
test_predictions = xgb.predict(X_test)

# Final Submission
submission = pd.DataFrame({
    'Item_Identifier': test['Item_Identifier'],
    'Outlet_Identifier': test['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})

submission.to_csv('../submission/submission.csv', index=False)