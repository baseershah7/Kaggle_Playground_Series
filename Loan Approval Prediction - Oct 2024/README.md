# Loan Approval Prediction - Oct 2024

## Overview
This project addresses the Kaggle Playground Series (October 2024) challenge focused on predicting loan approval status. The task is a binary classification problem where models determine whether a loan application should be approved based on applicant and loan attributes.

## Files
- `PSS4E10-(XGB, LGBM).ipynb`: Core notebook with data preprocessing, feature engineering, and model training using XGBoost and LightGBM.
- `PSS4E10-autogluon-automl-inference.ipynb`: Inference notebook for AutoGluon pre-trained model.
- `PSS4E10-autogluon-training.ipynb`: Training notebook for AutoGluon model with best-quality presets.
- `PSS4E10-catboost.ipynb`: CatBoost implementation with Optuna hyperparameter optimization.
- `PSS4E10-h2o-automl-inference.ipynb`: Inference notebook for H2O AutoML pre-trained model.
- `PSS4E10h2o-automl-training.ipynb`: Training notebook for H2O AutoML with leaderboard tracking.

## Data Sources
- **Competition Data**:
  - `/kaggle/input/playground-series-s4e10/train.csv`: Training data
  - `/kaggle/input/playground-series-s4e10/test.csv`: Test data for submission
  - `/kaggle/input/playground-series-s4e10/sample_submission.csv`: Submission template
- **External Data**:
  - `/kaggle/input/loan-approval-prediction/credit_risk_dataset.csv`: Additional loan risk data for training augmentation

## Approach
The solution employs multiple modeling strategies:

### 1. Data Preparation
- Combined competition data with external credit risk dataset
- Feature engineering created 24+ new features including:
  - Debt-to-income ratios
  - Employment stability flags
  - Risk categorization features
  - Interaction terms between demographic and financial attributes

### 2. Modeling Strategies
#### a) Traditional ML (XGBoost/LightGBM)
- Class-weighted training to handle imbalance
- Cross-validation with stratified KFold
- Out-of-fold (OOF) predictions for ensemble

#### b) AutoML Approaches
- **AutoGluon**:
  - Trained TabularPredictor with "best_quality" preset
  - Achieved 0.9582 validation AUC
- **H2O AutoML**:
  - Generated leaderboard of 32+ models
  - Best model: StackedEnsemble_AllModels (0.9612 AUC)

#### c) CatBoost Optimization
- Optuna hyperparameter tuning
- GPU acceleration
- Final model achieved 0.9678 validation AUC

## Models Used
| Approach          | Key Parameters                                                                 |
|-------------------|--------------------------------------------------------------------------------|
| XGBoost           | `scale_pos_weight=5`, `tree_method='hist'`                                     |
| LightGBM          | `class_weight='balanced'`, `categorical_feature` handling                     |
| CatBoost          | `learning_rate=0.02`, `depth=6`, `l2_leaf_reg=0.6`, `task_type='GPU'`         |
| AutoGluon         | `presets='best_quality'`, `max_models=30`, `time_limit=39600`                  |
| H2O AutoML        | `max_models=30`, `max_runtime_secs=39600`, `sort_metric='AUC'`                |

## Results
- **Best Validation AUC**: 0.9678 (CatBoost with Optuna tuning)
- **Submission Files**:
  - AutoGluon inference achieves ~0.9597 private score
  - H2O AutoML leader model achieves ~0.9613 private score
  - CatBoost optimized model achieves ~0.9678 validation AUC

## How to Reproduce
### 1. Install Dependencies
```bash
pip install autogluon.tabular h2o catboost optuna
```
### 2. Data Setup
Place all CSV files in /kaggle/input/ directories as shown in notebook paths.

### 3. Execution Order
- Train models using *-training.ipynb notebooks.
- Generate submissions using *-inference.ipynb notebooks.
  
## Key Insights
- Feature engineering combining multiple categorical variables significantly improved performance.
- Class weighting was critical due to 5:1 imbalance in loan approval status.
- AutoML approaches achieved competitive results with minimal tuning.
- CatBoost with GPU acceleration provided the best single-model performance.
- 
## Future Work
- Ensemble multiple AutoML approaches.
- Further hyperparameter tuning for XGBoost/LightGBM.
- Explore deep learning architectures.
- Analyze feature importance for business insights.
  
## Author
- [baseershah7](https://github.com/baseershah7)
