# BigMart Sales Prediction - Approach Note by Ritu Kukreja

## Motivation
From the beginning, I was excited about this challenge because predicting real-world retail sales is not just a modeling task — it’s an opportunity to understand customer behavior, product trends, and business operations.

I approached the BigMart Sales Prediction problem with curiosity, rigor, and enthusiasm to explore how product and outlet features influence sales!

## Exploratory Data Analysis (EDA)
The journey began with data inspection and visualization:

- Verified train/test dataset dimensions, feature types, and initial stats.

- Detected and addressed missing values in Item_Weight and Outlet_Size.

- Visualized key variable distributions such as Item_Outlet_Sales using histograms and boxplots.

- Explored categorical variables like Item_Fat_Content, Item_Type, and Outlet_Type.

- Uncovered inconsistencies (e.g., ‘LF’, ‘low fat’, ‘reg’) in Item_Fat_Content that required standardization.

These insights helped build strong data intuition and revealed how different item and outlet characteristics impact sales.

## Feature Engineering
To enhance predictive power:

- Standardized Item_Fat_Content by grouping inconsistent categories.

- Created Outlet_Years to capture outlet operational duration.

- Applied Label Encoding and One-Hot Encoding for categorical features.

- Filled missing Item_Weight with the mean and Outlet_Size with the mode.

- Combined train and test sets for consistent preprocessing and feature transformation.

This thoughtful feature engineering allowed the models to grasp underlying patterns in retail behavior.

## Modeling Journey
Tried a range of models to balance accuracy and complexity:

- Linear Regression: Simple and interpretable baseline.

- AdaBoost Regressor: Added some boosting power, moderate improvement.

- Gradient Boosting Regressor (XGBoost substitute): Best performing model on test split.

- Random Forest Regressor: Robust results, great at handling overfitting.

- Decision Tree Regressor and SVR: Tested for comparison, but with relatively higher RMSE.

Evaluation Metric: Root Mean Squared Error (RMSE) and R² Score were used consistently to assess performance on the validation set.

The final submission was generated using predictions from the Gradient Boosting Regressor.

## Reflections & Emotions
This project was a true rollercoaster 🎢. Sometimes models didn’t perform as expected, but experimentation and persistence kept me going. I realized how impactful even the smallest data cleaning steps or feature tweaks can be in real-world data science.

I thoroughly enjoyed the challenge, learned valuable lessons, and felt proud watching my RMSE improve with each iteration!

## Next Steps (If More Time)
- Hyperparameter tuning using Optuna or RandomizedSearchCV.

- Try more advanced models like LightGBM and CatBoost.

- Utilize SHAP values for interpretability and advanced feature selection.

"Every dataset has a story — and through this project, I loved being its storyteller." 📖✨

