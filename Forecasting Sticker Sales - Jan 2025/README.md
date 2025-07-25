# Forecasting Sticker Sales

## Overview
This folder contains a series of notebooks tackling the Kaggle Playground Series - Sticker Sales forecasting challenge. The goal is to predict sticker sales across different countries, stores, and products using a blend of classic feature engineering, boosting models, and automated tabular modeling frameworks.

## Notebooks

### 1. `PSS5E1-STICKER-SALES.ipynb`
- Feature engineering and boosting models for time series regression (LightGBM, CatBoost, XGBoost, Linear Regression).
- Rich temporal, categorical, and economic features (e.g., GDP ratios, cyclical encodings).
- Boxcox and quantile transformations for target normalization.
- Robust cross-validation (KFold, TimeSeriesSplit).
- Final predictions are transformed back to original scale and submitted.

### 2. `PSS5E1-autogluon.ipynb`
- Automated tabular regression using AutoGluon Tabular.
- Multiple model types: GBM, CatBoost, XGBoost, Neural Net, RF, Extra Trees, Linear Regression, KNN.
- 10-fold bagging, GPU acceleration, and advanced ensembling.
- Submission for sticker sales forecasting.

### 3. `PSS5E1-h2o-interferenc.ipynb`
- Inference pipeline using H2O’s AutoML trained XGBoost model.
- Loads best model and applies to test data.
- Handles feature engineering and time_id construction for proper prediction.
- Applies inverse Boxcox transformation to restore predictions to original sales scale.
- Produces final submission file.

### 4. `PSS5E1-h2o-training.ipynb`
- Training pipeline using H2O’s AutoML (XGBoost, GBM, DeepLearning, StackedEnsemble, DRF, GLM).
- Trains up to 50 models with extensive tabular features.
- Leaderboard and cross-validation OOF predictions for all models.
- Automated model saving and leaderboard export.
- Produces robust deep learning and ensemble models for sticker sales prediction.

## Data Sources
- `train.csv` and `test.csv`: Main sticker sales data.
- `sample_submission.csv`: Kaggle template.
- External: GDP features and other tabular enhancements.

## Project Workflow

1. **Data Loading & Exploration**
   - Reads train/test data, parses dates, visualizes target distributions.
   - Applies advanced transformations (log, sqrt, boxcox, Yeo-Johnson, quantile) to normalize and analyze sticker sales.

2. **Feature Engineering**
   - Creates time-based, cyclical, and economic features (year, month, week, day, GDP ratios).
   - Encodes categorical variables and addresses missing values and outliers.

3. **Modeling Approaches**
   - **Boosting Models:** LightGBM, CatBoost, XGBoost with rich feature sets.
   - **AutoML Frameworks:** AutoGluon for tabular regression, H2O for deep learning and ensembling.
   - **Model Ensembling:** Weighted ensembles combine predictions for robust accuracy.
   - **Inference Pipelines:** Efficient prediction and reverse transformation for final outputs.

4. **Evaluation & Results**
   - LightGBM, CatBoost, and XGBoost yield strong MAPE and R2 metrics.
   - AutoGluon and H2O ensembles achieve R2 > 0.97 on validation.
   - OOF predictions and model leaderboard exported for analysis.
   - Final predictions submitted in Kaggle-ready format.

5. **Submission**
   - All notebooks produce automated, robust submission files.
   - Pipelines handle prediction, transformation, and export for sticker sales forecasting.

## Key Insights
- Feature engineering (especially temporal and economic features) is crucial for high-accuracy forecasting.
- Advanced ensembling and AutoML frameworks (AutoGluon, H2O) outperform single models, especially on large, complex tabular data.
- Boxcox and quantile transformations help normalize skewed targets and improve model stability.
- Robust validation (KFold, TimeSeriesSplit, OOF predictions) supports leaderboard success.

## Future Work
- Add further feature engineering (holidays, promotions, weather).
- Hyperparameter tuning, custom stacking, and meta-modeling.
- Application to other time series forecasting tasks.

## How to Reproduce
1. Open each notebook in a Kaggle or Jupyter environment.
2. Ensure all required datasets are present in the specified directories.
3. Run all cells in sequence to preprocess, train, and generate submission files.

## Author
- [baseershah7](https://github.com/baseershah7)

