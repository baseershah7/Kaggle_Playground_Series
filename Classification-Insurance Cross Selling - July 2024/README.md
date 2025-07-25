
# Binary Classification - Insurance Cross Selling - July 2024

## Overview
This project is part of Kaggle's Playground Series (July 2024) and tackles binary classification for insurance cross-selling. The objective is to predict whether a policyholder will respond positively (1) or negatively (0) to cross-selling efforts, using a large, synthetic, highly imbalanced dataset.

## Files
- `PSS4E7_Optuna_EDA_>0.89.ipynb`: Main notebook containing all EDA, feature engineering, model training, hyperparameter optimization, and submission code.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s4e7/train.csv`: Training data.
- `/kaggle/input/playground-series-s4e7/test.csv`: Test data for predictions.
- `/kaggle/input/playground-series-s4e7/sample_submission.csv`: Submission template.
- `/kaggle/input/health-insurance-cross-sell-prediction-data/`: Original data.

## Dataset Description
- **Rows:** 11,504,798 instances
- **Features:** 12 columns (continuous, categorical, binary)
- **Target:** `Response` (binary, imbalanced 3:1)

| Feature            | Description                            |
|--------------------|----------------------------------------|
| Gender             | Policyholder gender                    |
| Age                | Policyholder age                       |
| Driving_License    | Has driving license                    |
| Region_Code        | Policyholder region                    |
| Previously_Insured | Was previously insured                 |
| Vehicle_Age        | Vehicle age category                   |
| Vehicle_Damage     | Vehicle damage indicator               |
| Annual_Premium     | Annual cost of insurance               |
| Policy_Sales_Channel| Sales channel                         |
| Vintage            | Days since policy issued               |
| Response           | Target: positive/negative response     |

## Project Workflow

### 1. Environment Setup
- Ensures correct versions of `scikit-learn` and `imbalanced-learn` for compatibility with sampling techniques.

### 2. Data Loading & Preprocessing
- Loads training and test data, sets appropriate dtypes.
- Maps categorical values to numeric codes.
- Creates new interaction features (e.g., `DrivingLicense_VehicleDamage`).
- Reduces memory usage for large-scale computations.

### 3. Exploratory Data Analysis (EDA)
- Distribution plots for response variable, continuous features, and categorical features.
- Pairplots and correlation heatmaps to visualize relationships and feature importance.
- Confirms high class imbalance and feature skewness.

### 4. Sampling for Imbalanced Data
- Tests various sampling strategies: SMOTE, NearMiss, random under/oversampling.
- Finds random oversampling and `class_weight='balanced'` most effective.

### 5. Feature Engineering
- New features created from domain knowledge and feature interactions.
- Ordinal and one-hot encoding for categorical variables.

### 6. Model Training & Evaluation
- Models: XGBoost, LightGBM, CatBoost, HistGradientBoosting, RandomForest, AdaBoost, ExtraTrees, GradientBoosting.
- Uses 3-fold K-Fold cross-validation for robust evaluation.
- Evaluation metric: ROC AUC.
- **Best CV Scores:**
    - XGBoost: Mean ROC AUC ~0.8894
    - CatBoost: Mean ROC AUC ~0.8708
    - LightGBM: Mean ROC AUC ~0.8792

### 7. Hyperparameter Optimization
- Uses Optuna to tune XGBoost hyperparameters (e.g., learning_rate, n_estimators, regularization).
- Final XGBoost model achieves ROC AUC > 0.89.

### 8. Feature Importance
- Visualizes feature importances for each model.

### 9. Submission
- Prepares test data using same preprocessing pipeline.
- Generates prediction probabilities with optimized XGBoost model.
- Writes predictions to `submission.csv` in required format.

## Key Findings
- **Highly imbalanced data**: Sampling and class weights crucial for performance.
- **Boosting models** outperform others on this dataset.
- **Feature engineering** and hyperparameter tuning (Optuna) contribute to achieving ROC AUC > 0.89.
- **Scaling** of features has minimal effect on boosting methods.

## How to Reproduce
1. Set up the Python environment as described in notebook prerequisites.
2. Open `PSS4E7_Optuna_EDA_>0.89.ipynb` in a Jupyter environment.
3. Run all cells sequentially, ensuring datasets are in the correct path.

## Future Work
- Explore model ensembling and stacking.
- Investigate additional synthetic features.
- Analyze leaderboard results and tune for generalization.

## Author
- [baseershah7](https://github.com/baseershah7)
