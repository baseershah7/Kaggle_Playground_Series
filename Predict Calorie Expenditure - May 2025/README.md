# Predict Calorie Expenditure - May 2025

## Overview
This project is part of Kaggle's Playground Series (May 2025) and focuses on predicting calorie expenditure using a synthetic dataset. The solution leverages ensemble methods, hyperparameter optimization, and feature engineering to achieve high predictive performance.

## Files
- `PSS5E5-CALORIE-ENSEMBLE.ipynb`: Main notebook for ensemble modeling, OOF predictions, and hill-climbing weight optimization.
- `PSS5E5-CALORIE-EXPENDITURE.ipynb`: Notebook for data loading, feature engineering, and generating model predictions.
- `PSS5E5-CALORIE-OPTUNA.ipynb`: Notebook for hyperparameter tuning using Optuna for CatBoost and XGBoost models.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s5e5/train.csv`: Training data.
- `/kaggle/input/playground-series-s5e5/test.csv`: Test data for predictions.
- `/kaggle/input/playground-series-s5e5/sample_submission.csv`: Submission template.
- `/kaggle/input/calorie-expenditure/`: Additional precomputed OOF predictions and CatBoost models.

## Dataset Description
| Feature       | Description                          |
|---------------|--------------------------------------|
| id            | Unique identifier                    |
| Sex           | Gender (male/female)                 |
| Age           | Participant age                      |
| Height         | Height in cm                         |
| Weight         | Weight in kg                         |
| Duration       | Exercise duration (minutes)          |
| Heart_Rate     | Average heart rate during exercise   |
| Body_Temp      | Body temperature (°C)                |
| Calories       | Target: Calories burned              |

## Project Workflow

### 1. Data Loading & Preprocessing
- Loaded training/test data and mapped categorical `Sex` to numeric values (0/1).
- Generated out-of-fold (OOF) predictions from pre-trained CatBoost and XGBoost models.

### 2. Ensemble Modeling
- Combined OOF predictions from **20 CatBoost** and **20 XGBoost** models.
- Optimized ensemble weights using **hill-climbing** to minimize RMSLE.
- Final predictions derived from weighted averaging of base models.

### 3. Feature Engineering
- Created interaction features (e.g., `bmi_hr` = BMI × Heart Rate).
- Implemented domain-specific proxies (e.g., metabolic rate estimates).

### 4. Hyperparameter Optimization
- Used **Optuna** to tune CatBoost/XGBoost hyperparameters:
  - Key parameters: `learning_rate`, `max_depth`, `subsample`, regularization terms.
  - GPU acceleration enabled for faster training.
- Achieved best RMSLE of **0.05942** with optimized XGBoost.

### 5. Model Training & Evaluation
- 10-fold cross-validation for robust evaluation.
- Metric: Root Mean Squared Logarithmic Error (RMSLE).

## Key Findings
- **Ensemble Superiority**: Weighted averaging of CatBoost/XGBoost OOF predictions outperformed single models.
- **Optuna Efficiency**: Bayesian optimization with delayed pruning significantly improved performance.
- **GPU Advantage**: Enabled rapid training of large ensembles.
- **Critical Features**: Duration, Heart Rate, and Body Temperature were most predictive.

## How to Reproduce
1. Enable GPU runtime in Kaggle.
2. Run notebooks (they are independent of each other ):
   - `PSS5E5-CALORIE-EXPENDITURE.ipynb` → Generate features/OOF predictions.
   - `PSS5E5-CALORIE-ENSEMBLE.ipynb` → Optimize ensemble weights.
   - `PSS5E5-CALORIE-OPTUNA.ipynb` → Tune hyperparameters.

## Future Work
- Include more diverse range of models for oof ensembling.
- Implement automated feature engineering pipelines.
- Investigate model stacking for further performance gains.

## Author
- [baseershah7](https://github.com/baseershah7)
