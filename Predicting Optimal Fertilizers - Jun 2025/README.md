# Predicting Optimal Fertilizers - Jun 2025

## Overview
This project tackles a 7-class classification problem to predict the optimal fertilizer for agricultural conditions using a synthetic dataset. The solution focuses on feature engineering, target encoding, and hyperparameter optimization to maximize Mean Average Precision at 3 (MAP@3).

## Files
- `PSS5E6-FEATURE-SELECTION.ipynb`: Implements hill-climbing feature selection and initial model exploration.
- `PSS5E6-OPTIMAL-FERTILIZERS.ipynb`: Contains data loading, target encoding, and final model training code.
- `PSS5E6-OPTUNA.ipynb`: Hyperparameter optimization notebook using Optuna for XGBoost.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s5e6/train.csv`: Main training data.
- `/kaggle/input/playground-series-s5e6/test.csv`: Test data for predictions.
- `/kaggle/input/playground-series-s5e6/sample_submission.csv`: Submission template.
- `/kaggle/input/fertilizer-prediction/`: Additional fertilizer prediction dataset.

## Dataset Description
| Feature       | Description                          |
|---------------|--------------------------------------|
| id            | Unique identifier                    |
| Temparature   | Ambient temperature (°C)            |
| Humidity      | Relative humidity (%)                |
| Moisture      | Soil moisture content (%)            |
| Soil Type     | Soil texture category                |
| Crop Type     | Cultivated crop variety              |
| Nitrogen      | Soil nitrogen content (kg/ha)        |
| Potassium     | Soil potassium content (kg/ha)       |
| Phosphorous   | Soil phosphorous content (kg/ha)     |
| Fertilizer Name| Target: 7 fertilizer classes        |

## Project Workflow

### 1. Data Loading & Preprocessing
- Combined training data with additional fertilizer dataset.
- Mapped categorical `Soil Type` and `Crop Type` to numeric codes.
- All features categorized as category ( continuous + categories included)
- Applied domain-specific feature engineering:
  - **Nutrient deficiency flags**: `N_deficient`, `P_deficient`, `K_deficient`
  - **Soil-crop interaction terms**: Encoded via ordinal encoding to capture unique relationships.

### 2. Feature Engineering
- **Target Encoding**: Implemented `IntenseTargetEncoder` with:
  - Multiple encoding strategies (mean, count, std, median, nunique)
  - Cross-validation to prevent data leakage
  - Noise injection to regularize rare categories
- **Artificial Features**:
  - NPK ratios and balance scores
  - Temperature-humidity indices
  - Soil-specific nutrient efficiency metrics

### 3. Feature Selection
- **Hill-Climbing Algorithm**:
  - Greedy forward selection optimizing traditional methods
  - Evaluated feature subsets using 5-fold cross-validation
  - Optimized for MAP@3 metric
- **Key Selected Features**:
  - Core soil metrics (Nitrogen, Phosphorous, Potassium)
  - Environmental factors (Temperature, Humidity, Moisture)
  - Engineered soil-crop interaction terms

### 4. Model Training
- **XGBoost Classifier** with GPU acceleration:
  - `tree_method='hist'`, `device='cuda'`
  - Categorical feature support via `enable_categorical=True`
- **Hyperparameter Tuning**:
  - Optuna optimization with TPESampler
  - Search space included:
    - Learning rate (0.01-0.3)
    - Max depth (3-30)
    - Regularization terms (reg_alpha, reg_lambda)
    - Category-specific parameters (max_cat_threshold, max_cat_to_onehot)

### 5. Evaluation
- **Metric**: Mean Average Precision at 3 (MAP@3)
- **Cross-Validation**: 5-fold stratified split
- **Best MAP@3**: 0.3017 (initial) → 0.3744 (optimized)

## Key Findings
1. **Feature Engineering Impact**:
   - Soil-crop interaction terms significantly improved performance
   - Target encoding outperformed raw categorical features

2. **Model Optimization**:
   - XGBoost outperformed CatBoost/LightGBM in preliminary tests
   - GPU acceleration enabled rapid hyperparameter search
   - Optimal parameters emphasized:
     - Moderate learning rate (0.01-0.03)
     - Deep trees (max_depth=21)
     - Category-specific thresholds (max_cat_threshold=4)

3. **Data Challenges**:
   - High cardinality categorical features required careful encoding
   - Converting both continuous and category dtypes as category has been the most impact for boosting models performance.
   - Class imbalance addressed through stratified sampling

## How to Reproduce
1. Enable GPU runtime in Kaggle.
2. Run notebooks :
   - `PSS5E6-FEATURE-SELECTION.ipynb` → Generate feature subsets
   - `PSS5E6-OPTIMAL-FERTILIZERS.ipynb` → Train final model
   - `PSS5E6-OPTUNA.ipynb` → Optimize hyperparameters

## Future Work
- Investigate ensemble methods combining XGBoost with neural networks and other boosting models for more diversity.

## Author
- [baseershah7](https://github.com/baseershah7)
