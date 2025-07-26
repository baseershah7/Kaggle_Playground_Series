# Loan Approval Prediction - Oct 2024

## Overview
This project addresses the Kaggle Playground Series (October 2024) challenge focused on predicting loan approval status using binary classification. The solution combines feature engineering, ensemble modeling, and AutoML approaches to achieve state-of-the-art results.

## Key Features
- Combines competition data with external credit risk dataset
- Implements 24+ engineered features including:
  - Debt-to-income ratios
  - Employment stability flags
  - Risk categorization features
  - Interaction terms between demographic/financial attributes
- Multiple modeling approaches including:
  - Traditional ML (XGBoost/LightGBM)
  - AutoML (AutoGluon/H2O)
  - Optimized CatBoost with Optuna tuning
- Ensemble learning with Out-of-Fold (OOF) predictions

## Files

| Notebook | Description |
|----------|-------------|
| `PSS4E10-(XGB, LGBM).ipynb` | Core notebook with feature engineering and traditional ML |
| `PSS4E10-autogluon-automl-inference.ipynb` | Inference for pre-trained AutoGluon model |
| `PSS4E10-autogluon-training.ipynb` | Training notebook for AutoGluon with best-quality presets |
| `PSS4E10-catboost.ipynb` | CatBoost implementation with Optuna hyperparameter optimization |
| `PSS4E10-h2o-automl-inference.ipynb` | Inference for pre-trained H2O AutoML model |
| `PSS4E10h2o-automl-training.ipynb` | Training notebook for H2O AutoML with leaderboard tracking |

## Data Sources

- **Competition Data**:
  - `/kaggle/input/playground-series-s4e10/train.csv`: Training data
  - `/kaggle/input/playground-series-s4e10/test.csv`: Test data
  - `/kaggle/input/playground-series-s4e10/sample_submission.csv`: Submission template

- **External Data**:
  - `/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv`: Additional loan risk data

## Approach

### 1. Data Preparation

- Combined competition data with external dataset
- Cleaned missing values and standardized formats
- Feature engineering created 24+ new features including:

```python
# Example feature engineering code
df['loan_to_income'] = (df['loan_amnt']/df['person_income'] - df['loan_percent_income'])
df['employment_stability'] = np.where(df['person_emp_length'] > 5, 'Stable', 'Unstable')
df['risk_flag'] = np.where((df['cb_person_default_on_file'] == 'Y') & 
                           (df['loan_grade'].isin(['C','D','E'])), 1, 0)
```

### 2. Modeling Strategies
#### a) Traditional ML (XGBoost/LightGBM)
- Class-weighted training to handle 5:1 class imbalance
- Stratified KFold cross-validation
- Out-of-fold (OOF) predictions for ensemble

#### b) AutoML Approaches
AutoGluon:

- Trained with presets='best_quality'
- Achieved 0.9582 validation AUC

H2O AutoML:

- Generated leaderboard of 32+ models
- Best model: StackedEnsemble_AllModels (0.9612 AUC)

#### c) CatBoost Optimization
- Optuna hyperparameter tuning with delayed pruning
- GPU acceleration
- Final model achieved 0.9678 validation AUC

``` python
# Key parameters from best trial
params = {
    'learning_rate': 0.00526,
    'max_depth': 10,
    'l2_leaf_reg': 1.08e-7,
    'border_count': 130,
    'scale_pos_weight': 3.0,
    'task_type': 'GPU'
}
```
### 3. Ensemble Learning
- Combined OOF predictions from multiple models
- Used hill climbing optimization to find optimal weights
- Final ensemble achieved best private score

## How to Reproduce
### 1. Install Dependencies
```bash
pip install autogluon.tabular h2o catboost optuna
```
### 2. Data Setup
Place all CSV files in /kaggle/input/ directories as shown in notebook paths.

### 3. Execute Order

Execute order not dependednt on each other.

## Key Insights
- Feature Engineering: Combining multiple categorical variables significantly improved performance
- Class Imbalance: Weighting strategies were critical due to 5:1 imbalance
- AutoML Efficiency: AutoML approaches achieved competitive results with minimal tuning
- GPU Acceleration: CatBoost with GPU achieved best single-model performance

## Future Work
- Ensemble multiple AutoML approaches
- Further hyperparameter tuning for XGBoost/LightGBM
- Explore deep learning architectures
- Analyze feature importance for business insights

## Author
- [baseershah7](https://github.com/baseershah7)
