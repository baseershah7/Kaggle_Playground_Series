# Regression with a Flood Prediction - May 2024

## Overview
This project tackles a regression task to predict flood probability using a synthetic dataset from Kaggle's Playground Series (May 2024). Two distinct approaches were implemented: a neural network and a polynomial feature-engineered LightGBM model.

## Files
- `PSS4E9_Neural.ipynb`: Neural network implementation using TensorFlow/Keras.
- `PSS4E9_Polynomial.ipynb`: Polynomial feature engineering with LightGBM.
- `README.md`: Project documentation.

## Data Sources
- `/kaggle/input/playground-series-s4e5/train.csv`: Training data.
- `/kaggle/input/playground-series-s4e5/test.csv`: Test data for predictions.
- `/kaggle/input/playground-series-s4e5/sample_submission.csv`: Submission template.
- `/kaggle/input/lgbm-poly3-robust/`: Pre-trained LightGBM model.

## Dataset Description
| Feature               | Description                          |
|-----------------------|--------------------------------------|
| MonsoonIntensity      | Intensity of monsoon                 |
| TopographyDrainage    | Drainage capability of topography   |
| RiverManagement       | River management effectiveness      |
| Deforestation         | Level of deforestation              |
| Urbanization          | Urban development level             |
| ClimateChange         | Impact of climate change            |
| DamsQuality           | Quality of dams                      |
| Siltation             | Silt accumulation                    |
| AgriculturalPractices | Farming practices                    |
| Encroachments         | Encroachment on water bodies        |
| ... (other features)  | ...                                  |
| FloodProbability      | Target: Probability of flooding     |

## Approach 1: Neural Network

### 1. Data Preprocessing
- Loaded training/test data, dropped `id` column.
- Split training data into train/validation (80/20).

### 2. Model Architecture
- **Sequential Model**:
  - Input layer: 20 features.
  - Hidden layers: 512 → 256 → 128 → 64 → 32 neurons, ReLU activation.
  - Output layer: 1 neuron, sigmoid activation (regression).
- **Optimizer**: SGD with learning rate 0.1.
- **Loss**: Mean Absolute Error (MAE).
- **Metrics**: R² score.

### 3. Training
- 10 epochs, batch size 128.
- Monitored validation loss without explicit early stopping.

### 4. Results
- Achieved validation R² of ~0.84 in final epoch.

### 5. Submission
- Predicted on test data, saved as `submission.csv`.

## Approach 2: Polynomial Features + LightGBM

### 1. Feature Engineering
- **Polynomial Features**: Degree 3 expansion using `PolynomialFeatures`.
- **Robust Scaling**: Applied to handle outliers.
- **Memory Management**: Processed data in chunks to avoid memory issues.

### 2. Model Training
- **LightGBM Regressor** with pre-trained weights (`lgbm_poly_3_robust.txt`).
- Key hyperparameters (from loaded model):
  - `n_estimators`: 2000
  - `learning_rate`: 0.012
  - `num_leaves`: 250
  - Regularization terms: `reg_alpha`, `reg_lambda`
  - GPU acceleration enabled.

### 3. Submission
- Loaded pre-trained model, predicted on test data, saved as `submission.csv`.

## Key Findings
- **Neural Network**: Achieved moderate performance with a simple architecture.
- **LightGBM + Polynomial Features**: Leveraged feature engineering and gradient boosting for potentially better results (exact score depends on LB).

## How to Reproduce
### Neural Network Approach:
1. Run `PSS4E9_Neural.ipynb` to train and submit.

### LightGBM Approach:
1. Ensure pre-trained model is in `/kaggle/input/lgbm-poly3-robust/`.
2. Run `PSS4E9_Polynomial.ipynb` to generate predictions.

## Future Work
- Experiment with hyperparameter tuning for the neural network.
- Explore more advanced feature engineering for the LightGBM model.
- Combine both approaches via ensemble methods or some try more diversity within same and different models for oof ensembling.

## Author
- [baseershah7](https://github.com/baseershah7)
