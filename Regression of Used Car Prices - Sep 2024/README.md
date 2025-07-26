# Regression of Used Car Prices - Sep 2024

## Overview
This project tackles a regression task to predict used car prices using a synthetic dataset from Kaggle's Playground Series (Sep 2024). The solution focuses on data cleaning, feature engineering, and model optimization to improve prediction accuracy.

## Files
- `PSS4E9_Preprocessing_Pipeline.ipynb`: Data understanding, missing value analysis and manual domain based-imputation, and initial preprocessing.
- `PSS4E9_cleaned_reg.ipynb`: Cleaned data version with updated preprocessing and model training.
- `PSS4E9_reg_raw.ipynb`: Raw data version showing original preprocessing steps.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s4e9/train.csv`: Training data.
- `/kaggle/input/playground-series-s4e9/test.csv`: Test data for predictions.
- `/kaggle/input/playground-series-s4e9/sample_submission.csv`: Submission template.

## Dataset Description
| Feature       | Description                          |
|---------------|--------------------------------------|
| id            | Unique identifier                    |
| brand         | Car manufacturer                     |
| model         | Car model                            |
| model_year    | Year of manufacture                  |
| milage        | Mileage (miles)                      |
| fuel_type     | Type of fuel (Gasoline, Hybrid, etc.)|
| engine        | Engine specifications                |
| transmission  | Transmission type                    |
| ext_col       | Exterior color                       |
| int_col       | Interior color                       |
| accident      | Accident history                     |
| clean_title   | Clean title status                   |
| price         | Target: Car price                    |

## Data Preprocessing

### 1. Missing Values Analysis
- **fuel_type**: 3% missing in training data, 2% in test. Imputed using group-wise mode (brand, model, model_year).
- **accident**: 2% missing. Left as-is due to non-relevance with other features.
- **clean_title**: 12% missing in training. Dropped due to minimal impact on model performance.

### 2. Feature Engineering
- **fuel_type Imputation**: Used group-wise mode imputation based on brand, model, and model_year to handle missing values.
- **Categorical Encoding**: Converted categorical features (brand, model, fuel_type, etc.) to ordinal codes using `OrdinalEncoder`.

### 3. Model Training
- **Model**: LightGBM Regressor with GPU acceleration.
- **Hyperparameters**:
  ```python
  lgbm_params = {
      'n_estimators': 1324,
      'num_leaves': 78,
      'max_depth': 21,
      'cat_smooth': 120,
      'learning_rate': 0.0146,
      'subsample': 0.57,
      'colsample_bytree': 0.5785,
      'min_split_gain': 0.3274,
      'min_child_weight': 68,
      'lambda_l2': 2.206e-06,
      'lambda_l1': 6.394e-05,
      'max_bin': 473
  }
  ```
## Key Findings
- Missing Value Handling: Imputing fuel_type using group-wise mode significantly improved model performance.
- Feature Impact: Dropping clean_title (due to minimal variance) and handling accident naively yielded better results.
- Model Choice: LightGBM outperformed other models (XGBoost, CatBoost) in preliminary tests, especially with GPU acceleration.
## How to Reproduce
- Environment Setup: Ensure GPU runtime is enabled in Kaggle.
- Run Notebooks:
-- PSS4E9_Preprocessing_Pipeline.ipynb → Analyze missing values and preprocess data.
-- PSS4E9_cleaned_reg.ipynb → Train the final LightGBM model.
- Submission: Use the preprocessed test data to generate predictions and submit.
## Future Work
- Explore advanced feature engineering (e.g., more interaction terms between brand, model, and engine).
- Experiment with model stacking or ensemble methods.
- Investigate hyperparameter tuning for further performance gains.
## Author
- [baseershah7](https://github.com/baseershah7)
