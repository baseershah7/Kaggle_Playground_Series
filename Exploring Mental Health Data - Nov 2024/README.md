# Exploring Mental Health Data - Nov 2024

## Overview
This project explores mental health survey data as part of the Kaggle Playground Series (November 2024). The goal is to predict depression status from a mix of demographic, academic, professional, and lifestyle features. The notebook achieved a top 2% leaderboard rank (35th place, top 1.3%).

## Files
- `PSS4E11-TOP-2%.ipynb`: Main notebook containing all data preprocessing, modeling, and submission code.
- `README.md`: Project documentation.

## Data Sources
- **train.csv**: Training data with features and target (`Depression`)
- **test.csv**: Test data for predictions
- **submission.csv**: Kaggle submission file

## Dataset Description

| Feature                      | Description                             |
|------------------------------|-----------------------------------------|
| id                           | Unique identifier                       |
| Name                         | Participant name                        |
| Gender                       | Gender                                  |
| Age                          | Age (years)                             |
| City                         | City of residence                       |
| Working Professional or Student | Status (Professional/Student)         |
| Profession                   | Job or field of study                   |
| Academic Pressure            | Level of academic pressure              |
| Work Pressure                | Level of work pressure                  |
| CGPA                         | Cumulative Grade Point Average          |
| Study Satisfaction           | Satisfaction with study                 |
| Job Satisfaction             | Satisfaction with job                   |
| Sleep Duration               | Typical sleep duration                  |
| Dietary Habits               | Type of diet                            |
| Degree                       | Degree obtained                         |
| Suicidal Thoughts            | History of suicidal thoughts            |
| Work/Study Hours             | Average daily work/study hours          |
| Financial Stress             | Level of financial stress               |
| Family History of Mental Illness | Family history                        |
| Depression                   | Target variable (binary)                |

- **Rows:** ~140,700
- **Features:** 19 (mix of categorical, continuous, binary)
- **Target:** `Depression` (0/1)

## Project Workflow

### 1. Data Loading & Preprocessing
- Converts all features except target to categorical type for efficient modeling.
- Handles missing data by treating all values as categories (string conversion).

### 2. Feature Engineering
- No manual feature engineering; relies on CatBoost’s native handling of categorical features.

### 3. Model Training
- Uses CatBoostClassifier with:
  - `iterations=4000`
  - `l2_leaf_reg=0.7`
  - `random_strength=0.7`
  - `learning_rate=0.0156`
  - `objective='Logloss'`
  - `eval_metric='AUC'`
- Trained on all available data, leveraging GPU acceleration for speed.

### 4. Prediction & Submission
- Test data processed identically to training data.
- Predictions generated for depression status.
- Submission file created in required Kaggle format.

## Results
- **Leaderboard rank:** 35th place (top 1.3%)
- **Model:** CatBoostClassifier (default + tuned parameters)
- **Approach:** Pure categorical modeling, no manual feature engineering required.

## How to Reproduce
1. Open `PSS4E11-TOP-2%.ipynb` in a Jupyter or Kaggle environment.
2. Ensure train/test datasets are available.
3. Run all cells sequentially to generate predictions and submission.

## Key Insights
- CatBoost is highly effective for large-scale categorical modeling.
- Minimal manual preprocessing needed; converting features to categorical is sufficient.
- Main finding is converting low-medium cardinality solid numerical features to category dtype massively increased generalization.
- Achieved top 2% leaderboard score with simple, robust pipeline.

## Future Work
- Explore feature engineering for further improvement.
- Test ensembling and stacking approaches.
- Analyze feature importances to understand drivers of depression.

## Author
- [baseershah7](https://github.com/baseershah7)

