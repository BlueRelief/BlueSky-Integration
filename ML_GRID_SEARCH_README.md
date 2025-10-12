# ML Model Grid Search - Hyperparameter Tuning

## Overview

This script performs comprehensive grid search cross-validation to find the best hyperparameters for each ML model used in disaster detection.

## Linear Issue

**Issue:** [BR-83 - Backend: Create Cross Grids for each model to actually find the best model possible](https://linear.app/bluerelief/issue/BR-83/backend-create-cross-grids-for-each-model-to-actually-find-the-best)

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install scikit-learn xgboost datasets psutil
```

### 3. Install OpenMP (macOS only, required for XGBoost)

```bash
brew install libomp
```

## Running the Grid Search

```bash
source venv/bin/activate
python ml_grid_search.py
```

This will take approximately 3-4 minutes to complete all grid searches.

## Models and Hyperparameter Grids

### 1. Logistic Regression
- **C**: [0.01, 0.1, 1.0, 10.0, 100.0]
- **penalty**: ['l1', 'l2']
- **solver**: ['liblinear', 'saga']
- **class_weight**: [None, 'balanced']
- **Total combinations**: 40

### 2. Naive Bayes
- **alpha**: [0.1, 0.5, 1.0, 2.0, 5.0]
- **fit_prior**: [True, False]
- **Total combinations**: 10

### 3. SVM
- **C**: [0.1, 1.0, 10.0, 100.0]
- **kernel**: ['linear', 'rbf']
- **gamma**: ['scale', 'auto']
- **class_weight**: [None, 'balanced']
- **Total combinations**: 32

### 4. Random Forest
- **n_estimators**: [50, 100, 200, 300]
- **max_depth**: [10, 20, 30, None]
- **min_samples_split**: [2, 5, 10]
- **min_samples_leaf**: [1, 2, 4]
- **max_features**: ['sqrt', 'log2', None]
- **class_weight**: [None, 'balanced']
- **Total combinations**: 864

### 5. XGBoost
- **n_estimators**: [50, 100, 200, 300]
- **max_depth**: [3, 5, 7, 10]
- **learning_rate**: [0.01, 0.05, 0.1, 0.2]
- **subsample**: [0.6, 0.8, 1.0]
- **colsample_bytree**: [0.6, 0.8, 1.0]
- **gamma**: [0, 0.1, 0.2]
- **min_child_weight**: [1, 3, 5]
- **Total combinations**: 5,184

## Results

The grid search produces the following output files:

### 1. `ml_grid_search_results.csv`
Raw results with best parameters and performance metrics for each model.

### 2. `ml_grid_search_results_formatted.csv`
Human-readable formatted results with percentages and units.

### 3. `ml_best_hyperparameters.json`
JSON file containing the best hyperparameters for each model. Use this to configure production models.

### 4. `ml_grid_search_comparison.csv`
Comparison between default hyperparameters and grid search optimized hyperparameters, showing improvement percentages.

## Key Findings

### Performance Improvements (F1-Score)

All models showed significant improvement after hyperparameter tuning:

1. **Naive Bayes**: +5.77% improvement
2. **Random Forest**: +5.28% improvement
3. **Logistic Regression**: +4.76% improvement
4. **SVM**: +4.76% improvement
5. **XGBoost**: +4.77% improvement

### Best Overall Model

**Naive Bayes** achieved the best performance:
- **CV F1-Score**: 99.88%
- **Test F1-Score**: 99.52%
- **Test Accuracy**: 99.52%
- **Grid Search Time**: 0.05s (fastest)
- **Best Parameters**:
  - alpha: 0.1
  - fit_prior: true

### Best Parameters by Model

#### Logistic Regression
```json
{
  "C": 100.0,
  "class_weight": null,
  "penalty": "l2",
  "solver": "saga"
}
```

#### Naive Bayes
```json
{
  "alpha": 0.1,
  "fit_prior": true
}
```

#### SVM
```json
{
  "C": 10.0,
  "class_weight": null,
  "gamma": "scale",
  "kernel": "linear"
}
```

#### Random Forest
```json
{
  "class_weight": null,
  "max_depth": null,
  "max_features": "log2",
  "min_samples_leaf": 1,
  "min_samples_split": 5,
  "n_estimators": 50
}
```

#### XGBoost
```json
{
  "colsample_bytree": 0.6,
  "gamma": 0,
  "learning_rate": 0.2,
  "max_depth": 10,
  "min_child_weight": 1,
  "n_estimators": 100,
  "subsample": 1.0
}
```

## Comparison with Default Parameters

| Model | Default F1 | Grid Search F1 | Improvement |
|-------|-----------|----------------|-------------|
| Naive Bayes | 94.09% | 99.52% | +5.77% |
| Random Forest | 94.53% | 99.52% | +5.28% |
| Logistic Regression | 95.00% | 99.52% | +4.76% |
| SVM | 95.00% | 99.52% | +4.76% |
| XGBoost | 94.53% | 99.04% | +4.77% |

## Computational Cost

| Model | Grid Search Time | Combinations Tested |
|-------|-----------------|---------------------|
| Naive Bayes | 0.05s | 10 |
| SVM | 0.42s | 32 |
| Logistic Regression | 3.53s | 40 |
| XGBoost | 90.52s | 5,184 |
| Random Forest | 101.59s | 864 |

## Recommendations

1. **For Production**: Use **Naive Bayes** with optimized parameters
   - Best performance (99.52% F1-score)
   - Fastest training time
   - Low memory footprint

2. **For Real-time Systems**: Use **Naive Bayes** or **Logistic Regression**
   - Both have prediction times < 1ms
   - Excellent accuracy

3. **For Maximum Accuracy**: Consider ensemble of top 3 models
   - Naive Bayes
   - Logistic Regression
   - SVM

## Next Steps

1. ✅ Implement grid search for all models
2. ✅ Compare with default parameters
3. ✅ Save best hyperparameters
4. 🔄 Update production models with optimized parameters
5. 🔄 Test on larger real-world dataset
6. 🔄 Implement ensemble model combining top performers

## Files Generated

- `ml_grid_search.py` - Main grid search script
- `ml_grid_search_results.csv` - Raw results
- `ml_grid_search_results_formatted.csv` - Formatted results
- `ml_best_hyperparameters.json` - Best parameters (JSON)
- `ml_grid_search_comparison.csv` - Default vs. optimized comparison
- `grid_search_output.log` - Full execution log

## Notes

- The script uses 5-fold cross-validation for all models
- F1-score (weighted) is used as the primary metric
- All random seeds are set to 42 for reproducibility
- The script automatically falls back to synthetic data if the real dataset fails to load

