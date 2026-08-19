import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load and Clean Dataset
print("Loading dataset...")
data_path = 'car_price.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# Extract brand from CarName
df['brand'] = df['CarName'].apply(lambda x: x.split()[0].lower())

# Correct brand spelling errors
brand_mapping = {
    'maxda': 'mazda',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'porcshce': 'porsche'
}
df['brand'] = df['brand'].replace(brand_mapping)

# Drop original CarName column (too high cardinality)
df = df.drop(columns=['CarName'])

# Separate Features and Target
X = df.drop(columns=['price'])
y = df['price']

# Step 2: Define Feature Types
categorical_cols = [
    'brand', 'carbody', 'drivewheel', 'enginelocation', 
    'fueltype', 'aspiration', 'doornumber', 'cylindernumber', 
    'enginetype', 'fuelsystem'
]
numerical_cols = [
    'symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 
    'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 
    'horsepower', 'peakrpm', 'citympg', 'highwaympg'
]

print(f"Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical.")

# Step 3: Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# Step 4: Define Preprocessing Pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Step 5: Define Models
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
}

# Step 6: Evaluate Models using 5-Fold Cross-Validation on the Train Set
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("\n--- Performing 5-Fold Cross-Validation ---")
for name, model in models.items():
    # Construct complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # R2 metric
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
    # MAE metric (sklearn returns negative MAE for cv scoring)
    mae_scores = -cross_val_score(pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    cv_results[name] = {
        'mean_r2': r2_scores.mean(),
        'std_r2': r2_scores.std(),
        'mean_mae': mae_scores.mean(),
        'std_mae': mae_scores.std()
    }
    print(f"{name:20} | Mean R2: {r2_scores.mean():.4f} (±{r2_scores.std():.4f}) | Mean MAE: ${mae_scores.mean():.2f}")

# Step 7: Train and Evaluate on Test Set, select Best Model
print("\n--- Evaluating Models on the Unseen Test Set ---")
best_model_name = None
best_test_r2 = -float('inf')
best_pipeline = None

test_results = {}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit on all training data
    pipeline.fit(X_train, y_train)
    
    # Predict on test data
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    test_results[name] = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'pred': y_pred
    }
    
    print(f"{name:20} | Test R2: {r2:.4f} | Test MAE: ${mae:.2f} | Test RMSE: ${rmse:.2f}")
    
    if r2 > best_test_r2:
        best_test_r2 = r2
        best_model_name = name
        best_pipeline = pipeline

print(f"\nBest Model Selected: {best_model_name} (Test R2: {best_test_r2:.4f})")

# Save Best Model Pipeline to Disk
model_filename = 'car_price_predictor.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(best_pipeline, f)
print(f"Saved best model pipeline to {model_filename}")

# Step 8: Generate and Save Diagnostic Plots
os.makedirs('plots', exist_ok=True)

# Colors and aesthetics setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
primary_color = '#1f77b4'
secondary_color = '#ff7f0e'
accent_color = '#2ca02c'

# Plot 1: Actual vs Predicted Prices (Test Set)
plt.figure(figsize=(8, 6))
y_pred_best = test_results[best_model_name]['pred']
plt.scatter(y_test, y_pred_best, color=primary_color, alpha=0.6, edgecolors='w', s=50, label='Predictions')
# Diagonal line (ideal predictions)
ideal_min = min(y_test.min(), y_pred_best.min())
ideal_max = max(y_test.max(), y_pred_best.max())
plt.plot([ideal_min, ideal_max], [ideal_min, ideal_max], color='red', linestyle='--', linewidth=2, label='Ideal Line')
plt.title(f'Actual vs. Predicted Car Price ({best_model_name})', fontsize=14, pad=15)
plt.xlabel('Actual Price ($)', fontsize=12)
plt.ylabel('Predicted Price ($)', fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=300)
plt.close()

# Plot 2: Residual Analysis
residuals = y_test - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs Predicted
axes[0].scatter(y_pred_best, residuals, color=secondary_color, alpha=0.6, edgecolors='w', s=50)
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
axes[0].set_title('Residuals vs. Predicted Prices', fontsize=13)
axes[0].set_xlabel('Predicted Price ($)', fontsize=11)
axes[0].set_ylabel('Residuals ($)', fontsize=11)

# Residuals Distribution Histogram
axes[1].hist(residuals, bins=20, color=primary_color, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
axes[1].set_title('Distribution of Residuals', fontsize=13)
axes[1].set_xlabel('Residual ($)', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)

plt.tight_layout()
plt.savefig('plots/residuals_analysis.png', dpi=300)
plt.close()

# Plot 3: Feature Importance (Only for Random Forest or Gradient Boosting if they win)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    try:
        # Extract features from preprocessor
        cat_encoder = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_cols))
        feature_names = numerical_cols + encoded_cat_features
        
        # Get importances
        importances = best_pipeline.named_steps['regressor'].feature_importances_
        
        # Create dataframe
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Keep top 15 features for readability
        top_feat_imp = feat_imp_df.head(15)
        
        plt.figure(figsize=(10, 6))
        plt.barh(top_feat_imp['Feature'][::-1], top_feat_imp['Importance'][::-1], color=accent_color, alpha=0.8)
        plt.title(f'Top 15 Feature Importance ({best_model_name})', fontsize=14, pad=15)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/feature_importance.png', dpi=300)
        plt.close()
        print("Generated feature importance plot.")
    except Exception as e:
        print(f"Could not generate feature importance plot: {e}")

# Step 9: Save training execution report to evaluation_results.md
with open('evaluation_results.md', 'w') as f:
    f.write(f"# Model Training & Evaluation Report\n\n")
    f.write(f"This report summarizes the performance of multiple machine learning models trained to predict car prices using `car_price.csv`.\n\n")
    
    f.write(f"## Cross-Validation Performance (Training Set)\n")
    f.write(f"We ran 5-Fold Cross-Validation on the training subset (80% of data) to evaluate and select the best model. The results are:\n\n")
    f.write(f"| Model | Mean R² Score | R² Std Dev | Mean MAE | MAE Std Dev |\n")
    f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")
    for name, res in cv_results.items():
        f.write(f"| {name} | {res['mean_r2']:.4f} | {res['std_r2']:.4f} | ${res['mean_mae']:.2f} | ${res['std_mae']:.2f} |\n")
    f.write(f"\n")
    
    f.write(f"## Unseen Test Set Performance\n")
    f.write(f"All models were evaluated on the remaining 20% of data (completely unseen during training/tuning):\n\n")
    f.write(f"| Model | Test R² Score | Test MAE | Test RMSE |\n")
    f.write(f"| :--- | :---: | :---: | :---: |\n")
    for name, res in test_results.items():
        f.write(f"| {name} | {res['r2']:.4f} | ${res['mae']:.2f} | ${res['rmse']:.2f} |\n")
    f.write(f"\n")
    
    f.write(f"### Final Selection: **{best_model_name}**\n")
    f.write(f"The best model is **{best_model_name}**, achieving a **Test R² score of {best_test_r2:.4f}** and a **Mean Absolute Error (MAE) of ${test_results[best_model_name]['mae']:.2f}**.\n\n")
    
    f.write(f"## Diagnostic Plots\n")
    f.write(f"The following plots have been generated in the `plots/` folder:\n")
    f.write(f"1. **`plots/actual_vs_predicted.png`**: Visual comparison of predicted prices vs true prices on the test set. Perfect predictions fall on the red dashed diagonal line.\n")
    f.write(f"2. **`plots/residuals_analysis.png`**: Analysis of residuals (errors) vs predicted values, along with a distribution histogram of prediction errors.\n")
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        f.write(f"3. **`plots/feature_importance.png`**: Visualizes the top 15 most important features that contribute to the price prediction.\n")

print("\nFinished training process. Report written to evaluation_results.md.")
