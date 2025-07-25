# Forecasting Sticker Sales

## Overview
This folder contains a series of notebooks tackling the Kaggle Playground Series - Sticker Sales forecasting challenge. The goal is to predict sticker sales across different countries, stores, and products using a blend of classic feature engineering, boosting models, AutoML frameworks, and deep learning approaches. The project explores advanced regression, time series feature engineering, and automated modeling pipelines.

## Notebooks

### 1. `PSS5E1-STICKER-SALES.ipynb`
- Feature engineering and boosting models for time series regression (LightGBM, CatBoost, XGBoost, Linear Regression).
- Rich temporal, categorical, and economic features (GDP ratios, cyclical time encodings).
- Boxcox and quantile transformations for target normalization.
- Robust cross-validation (KFold, TimeSeriesSplit).
- Final predictions are transformed back to original scale and submitted.

### 2. `PSS5E1-autogluon.ipynb`
- Automated tabular regression using AutoGluon Tabular.
- Multiple model types: GBM, CatBoost, XGBoost, Neural Net, RF, Extra Trees, Linear Regression, KNN.
- 10-fold bagging, GPU acceleration, and advanced ensembling.
- Leaderboard R2 scores > 0.97.
- Submission for both sticker sales and UHI Index prediction.

### 3. `PSS5E1-h2o-interferenc.ipynb`
- Inference pipeline using H2O’s AutoML trained XGBoost model.
- Loads best model and applies to test data.
- Handles feature engineering and time_id construction for proper prediction.
- Applies inverse Boxcox transformation to restore predictions to original sales scale.
- Produces final submission file.

### 4. `PSS5E1-h2o-training.ipynb`
- Training pipeline using H2O’s AutoML (XGBoost, GBM, DeepLearning, StackedEnsemble, DRF, GLM).
- Integrates additional external features (UHI Index, weather, building, geospatial, road network).
- Trains up to 50 models with 11,229 training rows and 26 features, using R2 as main metric.
- Leaderboard and cross-validation OOF predictions for all models.
- Automated model saving and leaderboard export.
- Produces robust deep learning and ensemble models for sticker sales and UHI prediction.

## Data Sources
- `train.csv` and `test.csv`: Main sticker sales data.
- `sample_submission.csv`: Kaggle template.
- Enhanced external datasets: GDP/UHI Index, weather, building, road network, and other geospatial features.

## Project Workflow

1. **Data Loading & Exploration**
   - Reads train/test data, parses dates, visualizes target distributions.
   - Applies advanced transformations (log, sqrt, boxcox, Yeo-Johnson, quantile) to normalize and analyze sticker sales.

2. **Feature Engineering**
   - Creates time-based, cyclical, economic, and geospatial features (year, month, week, day, GDP ratios, UHI Index).
   - Encodes categorical variables and addresses missing values and outliers.
   - GDP ratios and UHI Index add economic and climate context.

3. **Modeling Approaches**
   - **Boosting Models:** LightGBM, CatBoost, XGBoost with rich feature sets.
   - **AutoML:** AutoGluon for tabular regression, H2O for deep learning and ensembling.
   - **Model Stacking/Ensembling:** Weighted ensembles combine predictions for robust accuracy.
   - **Inference Pipelines:** Efficient prediction and reverse transformation for final outputs.

4. **Evaluation & Results**
   - LightGBM, CatBoost, and XGBoost yield strong MAPE and R2 metrics.
   - AutoGluon and H2O ensembles achieve R2 > 0.97 on validation.
   - OOF predictions and model leaderboard exported for analysis.
   - Final predictions submitted in Kaggle-ready format.

5. **Submission**
   - All notebooks produce automated, robust submission files.
   - Pipelines handle prediction, transformation, and export for both sticker sales and UHI Index tasks.

## Key Insights
- **Feature engineering** (especially temporal, economic, and geospatial features) is crucial for high-accuracy forecasting.
- **Advanced ensembling and AutoML frameworks** (AutoGluon, H2O) outperform single models, especially on large, complex tabular data.
- **Boxcox and quantile transformations** help normalize skewed targets and improve model stability.
- **Robust validation** (KFold, TimeSeriesSplit, OOF predictions) supports leaderboard success.

## Future Work
- Add further feature engineering (holidays, promotions, weather, climate indices).
- Hyperparameter tuning, custom stacking, and meta-modeling.
- Potential for more ensembling techniques.

## How to Reproduce
1. Open each notebook in a Kaggle or Jupyter environment.
2. Ensure all required datasets are present in the specified directories.
3. Run all cells in sequence to preprocess, train, and generate submission files.

## Author
- [baseershah7](https://github.com/baseershah7)
