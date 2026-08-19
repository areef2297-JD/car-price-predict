# Model Training & Evaluation Report

This report summarizes the performance of multiple machine learning models trained to predict car prices using `car_price.csv`.

## Cross-Validation Performance (Training Set)
We ran 5-Fold Cross-Validation on the training subset (80% of data) to evaluate and select the best model. The results are:

| Model | Mean R² Score | R² Std Dev | Mean MAE | MAE Std Dev |
| :--- | :---: | :---: | :---: | :---: |
| Ridge Regression | 0.8738 | 0.0897 | $1703.09 | $549.54 |
| Random Forest | 0.8687 | 0.0465 | $1655.04 | $298.06 |
| Gradient Boosting | 0.8445 | 0.0624 | $1736.71 | $347.91 |

## Unseen Test Set Performance
All models were evaluated on the remaining 20% of data (completely unseen during training/tuning):

| Model | Test R² Score | Test MAE | Test RMSE |
| :--- | :---: | :---: | :---: |
| Ridge Regression | 0.8872 | $2016.00 | $2915.87 |
| Random Forest | 0.9183 | $1841.23 | $2480.82 |
| Gradient Boosting | 0.9125 | $1861.64 | $2566.87 |

### Final Selection: **Random Forest**
The best model is **Random Forest**, achieving a **Test R² score of 0.9183** and a **Mean Absolute Error (MAE) of $1841.23**.

## Diagnostic Plots
The following plots have been generated in the `plots/` folder:
1. **`plots/actual_vs_predicted.png`**: Visual comparison of predicted prices vs true prices on the test set. Perfect predictions fall on the red dashed diagonal line.
2. **`plots/residuals_analysis.png`**: Analysis of residuals (errors) vs predicted values, along with a distribution histogram of prediction errors.
3. **`plots/feature_importance.png`**: Visualizes the top 15 most important features that contribute to the price prediction.
