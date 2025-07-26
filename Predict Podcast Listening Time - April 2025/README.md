# Predict Podcast Listening Time - April 2025

## Overview
This project is part of the Kaggle Playground Series (April 2025), focused on predicting podcast episode listening time in minutes. The challenge involves regression on a large and diverse dataset of podcast metadata, including both competition and external data sources.

## Files
- `PSS5E4_Listening_Time.ipynb`: Main notebook with all data processing, modeling, and submission steps.
- `train_data_pss5e4.csv`, `test_data_pss5e4.csv`: Enhanced train/test data exports (with engineered features).
- `submission.csv`: Kaggle submission file.

## Data Sources
- `/kaggle/input/playground-series-s5e4/train.csv`: Competition training data.
- `/kaggle/input/playground-series-s5e4/test.csv`: Competition test data.
- `/kaggle/input/playground-series-s5e4/sample_submission.csv`: Submission template.
- `/kaggle/input/podcast-listening-time-prediction-dataset/podcast_dataset.csv`: Additional podcast data for enrichment.

## Dataset Description
**Features include:**
- Podcast_Name
- Episode_Title
- Episode_Length_minutes
- Genre
- Host_Popularity_percentage
- Publication_Day
- Publication_Time
- Guest_Popularity_percentage
- Number_of_Ads
- Episode_Sentiment
- [Many combinations of categorical features and their target/count encodings]
- **Target:** Listening_Time_minutes (continuous)

## Project Workflow

### 1. Data Loading & Combination
- Loads both competition (synthetic) and original podcast datasets.
- Concatenates, deduplicates, and drops rows with missing target values.
- Final training set: ~795,000 rows, 163 features.

### 2. Data Exploration
- Visualizes listening time distribution with KDE/histograms.
- Identifies feature types and missing values.

### 3. Feature Engineering
- Creates interaction features by combining categorical columns (2- to 4-way combos).
- Applies target encoding and count encoding for categorical features with high cardinality.
- Handles missing values and rare categories.

### 4. Outlier Removal
- Outlier filtering available via IQR-based method for numeric features.

### 5. Model Training
- Uses XGBoost for regression (`XGBRegressor`).
- Handles categorical features via string conversion and categorical dtype.
- Trains on the full enhanced feature set.
- Cross-validation and other models (ExtraTrees, CatBoost, LightGBM, HistGradientBoosting) available in code but not used in final submission.

### 6. Prediction & Submission
- Prepares test data with same feature engineering pipeline.
- Generates predictions for listening time in minutes.
- Writes results to `submission.csv` for Kaggle.

## Results
- Example predictions range from ~6 minutes to ~82 minutes per episode.
- Model: XGBoost with histogram tree method and categorical feature support.
- Feature importance analysis available in notebook for further exploration.

## How to Reproduce
1. Place all required datasets in the correct input directories.
2. Open `PSS5E4_Listening_Time.ipynb` in Kaggle or Jupyter environment.
3. Run all cells sequentially to process data, engineer features, train the model, and generate submissions.

## Key Insights
- Combining original podcast data with given competition data increases training diversity and improves model robustness.
- Interaction features and target/count encoding capture deep relationships in podcast metadata.
- Handling categorical data and missing values is crucial for performance.
- XGBoost with categorical handling is effective for large, mixed-type tabular regression.

## Future Work
- Explore deep learning and ensembling approaches.
- Apply hyperparameter tuning for further improvements.
- Add time-based features (e.g., episode recency, trends).
- Analyze feature importances and interpretability.

## Author
- [baseershah7](https://github.com/baseershah7)
