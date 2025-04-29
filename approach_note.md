# BigMart Sales Prediction - Approach Note by Ritu Kukreja

## Motivation
From the beginning, I was excited about this challenge because predicting real-world retail sales is not just a modeling task — it’s an opportunity to understand customer behavior, product trends, and business operations.

I approached the BigMart Sales Prediction problem with curiosity, rigor, and a lot of enthusiasm to explore how product and outlet features influence sales!

## Exploratory Data Analysis (EDA)
I started with deep exploration:

- Checked data dimensions, types, and missing values.

- Visualized distributions of key variables like Item_Outlet_Sales and relationships with categorical features.

- Identified:

    - Missing values in Item_Weight and Outlet_Size

    - Zero values in Item_Visibility

    - Inconsistencies in Item_Fat_Content

This phase revealed the "story" hidden in the data, which made me even more invested in building meaningful features!

## Feature Engineering
Realizing that "raw" features weren’t enough, I created new variables to capture deeper patterns:

- Outlet_Age: Older outlets may have loyal customers and stable sales.

- Item_Visibility_MeanRatio: Corrected skewed zero visibility.

- Item_Category: Grouped items broadly into Food, Drinks, Non-Consumables.

- Standardized Item_Fat_Content: To fix text inconsistencies (LF, low fat, etc.)

Each transformation was guided by a simple question in my mind:
"Can this feature give the model a better understanding of customer buying behavior?"

## Modeling Journey
I adopted a two-phase strategy:

- Phase 1: Quick baseline with RandomForestRegressor → RMSE evaluated.

- Phase 2: After thorough feature engineering, switched to a more powerful XGBoost Regressor → Achieved improved RMSE.

Along the way, I constantly balanced model complexity vs interpretability.
Every improvement, even a small drop in RMSE, felt like a victory!

## Reflections & Emotions
This project was a true rollercoaster 🎢 — sometimes the models didn’t perform as expected, but experimentation and persistence kept me going.
Every data cleaning step, every feature created, every model tuned — made me realize how impactful small details are in real-world data science.

I thoroughly enjoyed the challenge, learned valuable lessons, and felt proud watching my RMSE improve!

## Next Steps (If More Time)
- Hyperparameter tuning with Optuna/RandomSearchCV.

- Try LightGBM and CatBoost for further boosting.

- Advanced feature selection based on SHAP values.

“Every dataset has a story — and through this project, I loved being its storyteller.” 📖✨

