# 🛒 BigMart Sales Prediction Challenge

## 📖 Project Overview
This repository contains my solution for the BigMart Sales Prediction challenge.

The goal is to predict product sales across various outlets using historical sales and outlet/product attributes. The workflow spans from initial data inspection to advanced modeling using multiple regression techniques.

## 🛠️ Repository Structure
| Folder/File | Description |
| :--- | :--- |
| `data\` | Contains train.csv and test.csv datasets. |
| `notebooks\EDA.ipynb` | Complete data exploration, feature engineering notebook, Model building, evaluation, and final prediction script. |
| `submission\submission.csv` | Final prediction file ready for submission. |
| `approach_note.md` | 1-page document explaining my thought process and modeling approach. |
| `screenshot/` | Screenshot of the best leaderboard rank and score. |
| `README.md` | This file — overview of the project. |

## 🔍 Key Steps Followed
- 📊 Data Exploration & Cleaning

    - Assessed missing values and distributions.

    - Visualized categorical and numerical features.

    - Treated outliers and inconsistent values (e.g., Item_Fat_Content).

- 🛠️ Feature Engineering

    - Created Outlet_Years from establishment year.

    - Imputed missing values in Item_Weight and Outlet_Size.

    - Standardized categorical values.

    - One-hot encoded categorical features.

- 🤖 Modeling Techniques Used

    - Linear Regression

    - AdaBoost Regressor

    - Gradient Boosting Regressor

    - Random Forest Regressor

    - Decision Tree Regressor

    - Support Vector Regressor (SVR)

- 📈 Evaluation

    - Models were validated using RMSE and R² score.

    - Gradient Boosting and Random Forest performed best based on evaluation metrics.

- 🚀 Submission

    - Predictions on the test set were generated using the best model.

    - Final predictions were exported to submission.csv.

## 💬 Reflections
This project deepened my understanding of end-to-end regression modeling and the critical impact of data preprocessing and feature engineering. I experimented with several models to understand their behavior on retail sales data.

“In data science, curiosity and persistence are our greatest assets.” ✨

## 📚 Technologies Used
- Python: Pandas, NumPy, Matplotlib, Seaborn

- Scikit-learn: Regression models and preprocessing

- XGBoost: Advanced gradient boosting

- Jupyter Notebook: For experimentation and visualization

## 🙋‍♂️ About Me
[Ritu Kukreja](https://www.linkedin.com/in/ds-rvk/)
Data Scientist passionate about solving real-world problems with data, creativity, and perseverance.

# ⭐ Thank you for visiting!
