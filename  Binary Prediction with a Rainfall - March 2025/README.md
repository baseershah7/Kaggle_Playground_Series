# Binary Prediction with a Rainfall - March 2025

## Overview
This project is part of the Kaggle Playground Series (March 2025) and focuses on binary classification to predict rainfall occurrence based on meteorological features. Multiple datasets and a variety of machine learning models are used to build, evaluate, and ensemble predictions.

## Files
- `PSS5E3.ipynb`: Main notebook containing all data analysis, feature engineering, modeling, and submission code.
- `README.md`: Documentation for the project.

## Data Sources
- `/kaggle/input/playground-series-s5e3/train.csv`: Main training data.
- `/kaggle/input/playground-series-s5e3/test.csv`: Main test data.
- `/kaggle/input/playground-series-s5e3/sample_submission.csv`: Sample submission template.
- `/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv`: Original data as the main data from the episodes are synthetic.

## Project Workflow

### 1. Data Loading & Exploration
- Loads multiple CSVs: primary and supplementary datasets.
- Cleans column names and ensures consistent data types.
- Merges supplementary rainfall data for feature enhancement.
- Checks for missing values and performs basic exploratory data analysis.

### 2. Feature Engineering
- Creates artificial features such as:
  - Rolling means and variances for temperature and windspeed.
  - Monsoon season indicator.
  - Temperature and wind direction transformations.
  - Multiple interaction and difference features.
- Handles missing values and scales features using `StandardScaler`.

### 3. Model Training & Evaluation
- Uses several classifiers:
  - ExtraTreesClassifier
  - RandomForestClassifier
  - XGBoost
  - CatBoost
  - LightGBM
  - Logistic Regression
  - KNN
  - HistGradientBoosting
  - MLPClassifier
- Cross-validation performed using `StratifiedKFold`.
- Main evaluation metric: ROC-AUC.
- Example scores (ExtraTreesClassifier, 5-fold CV):  
  - Fold scores: 0.897, 0.876, 0.907, 0.886, 0.924  
  - Mean: ~0.898

### 4. Model Ensembling
- Implements rank-based ensemble blending for multiple classifier outputs.
- Optimizes ensemble weights using Bayesian optimization and hill climbing.
- Converts model ranks to probability scores for final prediction.

### 5. Feature Selection
- Uses RFECV for optimal feature subset selection.

### 6. Submission
- Generates predictions for the test set and writes submission file.

## Results
- Best single-model ROC-AUC (CV): ~0.898 (ExtraTreesClassifier).
- Ensemble methods further explored for improved robustness.
- Submission file created for Kaggle evaluation.

## How to Reproduce
1. Open `PSS5E3.ipynb` in a Jupyter environment (Kaggle(recommended), VSCode, etc.).
2. Ensure datasets are available in appropriate input directories.
3. Run all cells sequentially to reproduce the analysis, modeling, and submission steps.

## Key Insights
- Feature engineering (especially rolling statistics and interaction terms) significantly improves model performance.
- Ensemble blending and rank transformation techniques provide robustness against overfitting and boost leaderboard scores.
- Adding Original dataset to the given synthetic one's increase performance.

## Future Work
- Experiment with additional more advanced ensemble strategies.
- Explore more aggressive feature selection and dimensionality reduction.
- Fine-tune hyperparameters for all models.

## Author
- [baseershah7](https://github.com/baseershah7)

