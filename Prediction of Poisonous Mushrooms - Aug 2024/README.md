# Prediction of Poisonous Mushrooms - Aug 2024

## Overview
This project addresses the Kaggle Playground Series Season 4, Episode 8 challenge. The goal is to perform binary classification to predict whether a mushroom is poisonous ('p') or edible ('e') based on a large synthetic dataset containing a mixture of continuous and categorical features, many of which have significant missing data.

## Files
- `PSS4E8_Poisonous_Mushrooms.ipynb`: Main notebook containing all steps from data loading, exploration, preprocessing, modeling, and submission generation.
- `submission.csv`: Final predictions file formatted for submission to Kaggle.

## Data Sources
- `/kaggle/input/playground-series-s4e8/train.csv`: Competition training data.
- `/kaggle/input/playground-series-s4e8/test.csv`: Competition test data.
- `/kaggle/input/playground-series-s4e8/sample_submission.csv`: Sample submission file template.

## Dataset Description
The dataset contains over 3.1 million instances with 22 features (including the 'id' column).

**Features include:**
- **Continuous (float64):**
  - `cap-diameter`: Diameter of the mushroom cap.
  - `stem-height`: Height of the mushroom stem.
  - `stem-width`: Width of the mushroom stem.
- **Categorical (object):**
  - `cap-shape`, `cap-surface`, `cap-color`, `gill-attachment`, `gill-spacing`, `gill-color`, `stem-root`, `stem-surface`, `stem-color`, `veil-type`, `veil-color`, `ring-type`, `spore-print-color`, `habitat`, `season`.
- **Binary Categorical (object):**
  - `does-bruise-or-bleed`: Whether the mushroom bruises or bleeds.
  - `has-ring`: Whether the mushroom has a ring.
- **Target:**
  - `class`: The classification label ('e' for edible, 'p' for poisonous).

**Key Characteristics:**
- **Size:** ~3.1 million rows.
- **Balance:** Dataset is relatively balanced (~45% poisonous, ~55% edible).
- **Missing Data:** Several categorical features have significant amounts of missing data (e.g., `cap-surface`, `gill-attachment`, `stem-root`, `veil-type`, `spore-print-color`).
- **Cardinality:** High cardinality in nominal categorical features.
- **Distribution:** Continuous features (`cap-diameter`, `stem-height`, `stem-width`) are highly skewed.

## Project Workflow

### 1. Data Loading & Initial Inspection
- Loads training and test datasets from CSV files.
- Drops the `id` column from the training set for analysis.
- Performs initial inspection of data structure, value counts, data types, and calculates missing value percentages for each feature.

### 2. Exploratory Data Analysis (EDA)
- **Missing Values:** Visualizes missing data patterns using a heatmap.
- **Target Distribution:** Plots the distribution of the `class` variable using count and pie plots to confirm balance.
- **Continuous Features:** Analyzes the distribution of `cap-diameter`, `stem-height`, and `stem-width` with respect to `class` using histograms/KDEs, violin plots, and Q-Q plots. Calculates descriptive statistics, skewness, and kurtosis.
- **Categorical Features:** Examines the distribution of all categorical features w.r.t `class` using count and pie plots.
- **Correlations:** Displays a correlation heatmap for continuous features, showing moderate positive correlations between `cap-diameter`/`stem-width` and `stem-height`/`stem-width`.
- **Pairwise Plots:** Generates a pairplot for continuous features colored by class.

### 3. Data Preprocessing
- Identifies categorical columns (excluding target) and continuous columns.
- Encodes categorical features using `OrdinalEncoder` with `handle_unknown='use_encoded_value'` and `unknown_value=-1`. This handles both known categories and potential unseen categories in the test set or future data.
- Maps the target variable `class` from ('e', 'p') to (0, 1) for model training.

### 4. Model Selection & Validation
- Defines the evaluation metric: `Matthews Correlation Coefficient (MCC)` using `make_scorer`.
- Selects several gradient boosting models for comparison:
    - XGBoost (`XGBClassifier` - using histogram method and CUDA)
    - LightGBM (`LGBMClassifier` - using GPU)
    - Scikit-learn Histogram Gradient Boosting (`HistGradientBoostingClassifier`)
    - CatBoost (`CatBoostClassifier`)
- Performs 3-Fold Cross-Validation using `KFold` and `cross_val_score`.
- Compares model performance based on mean MCC scores and standard deviation across folds.
- **Results:** XGBoost (MCC ~0.9828) and CatBoost (MCC ~0.9825) outperform LightGBM and HistGradientBoosting (MCC ~0.9784).

### 5. Feature Importance Analysis
- Trains the top-performing models (specifically excluding HistGradient and CatBoost from the visualization step shown in the notebook).
- Visualizes feature importance derived from the trained models (e.g., XGBoost) to understand which features contribute most to the predictions.

### 6. Final Model Training, Prediction & Submission
- Selects XGBoost as the final model based on cross-validation performance.
- Trains the selected XGBoost model (`XGBClassifier`) on the *entire* preprocessed training dataset (`X`, `y`).
- Applies the same fitted `OrdinalEncoder` to preprocess the test set features.
- Generates predictions on the preprocessed test set using the trained XGBoost model.
- Maps the predicted labels back from (0, 1) to ('e', 'p').
- Constructs the final submission DataFrame with `id` and `class` columns.
- Saves the predictions to `submission.csv` in the required Kaggle format.

## Results
- Achieved high cross-validation MCC scores (>0.98) using gradient boosting models, indicating strong model performance.
- Final model used for submission: XGBoost (`XGBClassifier`).
- Feature importance analysis highlighted key predictors such as `gill-color`, `spore-print-color`, `cap-color`, and `stem-width`.

## How to Reproduce
1. Ensure the required Kaggle datasets are available in the `/kaggle/input/playground-series-s4e8/` directory.
2. Open `PSS4E8_Poisonous_Mushrooms.ipynb` in a Kaggle or compatible Jupyter environment (with necessary libraries like XGBoost, LightGBM, CatBoost, scikit-learn, pandas, numpy, matplotlib, seaborn).
3. Run all cells sequentially to execute the complete workflow and generate the `submission.csv` file.

## Key Insights
- Gradient boosting models (XGBoost, LightGBM, CatBoost) are highly effective for this large-scale, tabular classification task with mixed data types.
- Ordinal encoding is a suitable and effective strategy for handling the high-cardinality categorical features in this dataset.
- Despite significant missing data in several features, tree-based models like XGBoost can leverage this information effectively without explicit imputation.
- The Matthews Correlation Coefficient is a robust metric for evaluating performance, especially in cases where precision for both classes is important.
- Hyperparameter tuning (e.g., using Optuna as suggested by the notebook author) is a logical next step for potential performance improvement.

## Future Work
- Implement hyperparameter tuning (e.g., using Optuna or similar libraries) for the XGBoost model to potentially increase the MCC score.
- Experiment with alternative missing data handling strategies (e.g., imputation methods, adding explicit missing indicators).
- Try different encoding techniques for categorical variables (e.g., Target Encoding, Leave-One-Out Encoding, potentially after handling leakage).
- Explore ensemble methods combining the top-performing models (XGBoost, CatBoost).
- Investigate the specific impact of high-cardinality features and the effectiveness of ordinal encoding versus other methods.

## Author
- [baseershah7](https://github.com/baseershah7)

