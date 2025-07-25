# Backpack Prediction Challenge - Feb 2025

## Overview
This project is part of the Kaggle Playground Series (February 2025) and aims to predict backpack prices using product features such as brand, material, size, and more. The challenge is a regression task and leverages modern ensemble machine learning techniques.

## Files
- `PSS5E2.ipynb`: Main notebook containing all data analysis, feature engineering, model training, prediction, and submission code.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s5e2/train.csv`: Primary training data -- Synthetic.
- `/kaggle/input/playground-series-s5e2/test.csv`: Test data for prediction -- Synthetic.
- `/kaggle/input/playground-series-s5e2/sample_submission.csv`: Sample submission template for Kaggle.
- `/kaggle/input/playground-series-s5e2/training_extra.csv`: Original dataset.

## Project Workflow

### 1. Data Loading & Exploration
- Loads all relevant CSVs for the challenge.
- Identifies categorical features and converts them to the appropriate data types.
- Visualizes distribution of target variable (`Price`) using Seaborn's KDE plot.

### 2. Data Preprocessing
- Handles categorical variables by converting them to string (with missing value imputation) then to category dtype.
- Ensures test set matches training set feature encoding.

### 3. Model Training
- Uses CatBoostRegressor for its native handling of categorical features and strong performance in tabular data tasks.
- Trains the model on all provided training data.
- No explicit hyperparameter tuning shown (defaults used).

### 4. Prediction & Submission
- Predicts `Price` for the test data using the trained model.
- Writes predictions to `submission.csv` in the required Kaggle format.
- Displays the sample submission head for verification.
- Visualizes predicted prices using KDE plot.

## Results
- Model: CatBoostRegressor (default settings)
- Example predictions (first five):
    | id     | Price      |
    |--------|------------|
    | 300000 | 87.71      |
    | 300001 | 80.71      |
    | 300002 | 93.52      |
    | 300003 | 84.14      |
    | 300004 | 80.85      |
- Performance metrics (e.g., RMSE, MAE) not shown in notebook.

## How to Reproduce
1. Open `PSS5E2.ipynb` in a Jupyter environment (Kaggle, VSCode, etc.).
2. Ensure datasets are available in the specified input directories.
3. Run all cells sequentially to reproduce the analysis, modeling, and submission steps.

## Key Insights
- CatBoost simplifies handling of categorical features and delivers solid baseline performance.
- Visualization aids in understanding target and prediction distributions.
- Supplementary data (`training_extra.csv`) is loaded but not explicitly used in modeling; potential for further feature engineering.

## Future Work
- Hyperparameter tuning for CatBoost or alternative models (XGBoost, LightGBM).
- Feature engineering using supplementary data.
- Model stacking or ensembling to boost performance.
- Evaluation of predictions on validation set (cross-validation, error metrics).

## Author
- [baseershah7](https://github.com/baseershah7)


