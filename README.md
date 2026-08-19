# Car Price Predictor Dashboard 🚗

An interactive Machine Learning project that predicts car prices based on vehicle specifications. It includes:
1. **Model Training & Comparison Script (`train.py`)**: Preprocesses the dataset and evaluates Ridge Regression, Random Forest, and Gradient Boosting Regressors.
2. **Interactive CLI prediction tool (`predict.py`)**: Allows users to make predictions from the terminal using command line parameters or an interactive prompt.
3. **Responsive Glassmorphic Web Dashboard (`index.html`)**: A client-side web application running real-time ML predictions in the browser with no backend server required.

---

## 📊 Model Performance

We trained and evaluated the models on `car_price.csv` (80-20 train-test split). Performance on the unseen test set is summarized below:

| Algorithm Candidate | Test R² Score | Test MAE | Test RMSE | Usage |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | **91.83%** | **$1,841.23** | **$2,480.82** | Local Python CLI (`predict.py`) |
| **Ridge Regression** | **88.72%** | **$2,016.00** | **$2,915.87** | Browser Web UI (`index.html`) |
| Gradient Boosting | 91.25% | $1,861.64 | $2,566.87 | Evaluation Baseline |

---

## 🛠️ Tech Stack & Key Features

- **Modeling & Preprocessing**: Python, Pandas, NumPy, Scikit-Learn (ColumnTransformer, StandardScaler, OneHotEncoder).
- **Web UI Layout**: Vanilla HTML5 (semantic layout, custom responsive range sliders, dynamic tab menus).
- **Web UI Styling**: Custom dark theme with glassmorphic cards (`backdrop-filter`) and glowing CSS neon gradients.
- **Serverless Predictor**: We exported the Ridge regression coefficients, scaler scaling factors, and one-hot encoding dictionaries into `model_data.js`. The calculations run in pure client-side JavaScript in **< 1 millisecond** with no backend.
- **Dynamic Options**: Dropdowns in the Web UI are populated automatically from the model's categorical vocabulary.
- **Diagnostic Visualizations**: Includes diagnostic plots inside the dashboard (`plots/`).

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
Ensure you have Python 3 and Scikit-Learn installed:
```bash
pip install pandas numpy scikit-learn matplotlib
```

### 2. Run Python Predictions
You can predict the price of a car using the CLI:
```bash
# Get predictions interactively
python predict.py

# Or pass specific parameters
python predict.py --brand bmw --horsepower 200 --enginesize 180 --curbweight 3000
```

### 3. Run Web Predictor Locally
Simply open the `index.html` file in any modern web browser.
- Adjust sliders (Horsepower, Engine Size, Curb Weight, MPG).
- Select options (Brand, Body Style, Drive Wheel, Fuel Type).
- Watch the estimated price and feature contributions update dynamically!

---

## 📈 Diagnostic Plots

All plots are generated during training and saved to the `plots/` folder:
- **`actual_vs_predicted.png`**: Actual vs. predicted prices on the test split.
- **`residuals_analysis.png`**: Residual values and error distribution.
- **`feature_importance.png`**: Top 15 variables influencing vehicle MSRP.
