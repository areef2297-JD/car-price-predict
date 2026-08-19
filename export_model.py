import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

print("Loading data...")
df = pd.read_csv('car_price.csv')

# Clean brand
df['brand'] = df['CarName'].apply(lambda x: x.split()[0].lower())
brand_mapping = {
    'maxda': 'mazda', 'toyouta': 'toyota', 'vokswagen': 'volkswagen',
    'vw': 'volkswagen', 'porcshce': 'porsche'
}
df['brand'] = df['brand'].replace(brand_mapping)
df = df.drop(columns=['CarName'])

X = df.drop(columns=['price'])
y = df['price']

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

print("Preprocessing and fitting Ridge model...")
# Preprocessing pipelines
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

X_preprocessed = preprocessor.fit_transform(X)
scaler = preprocessor.named_transformers_['num']
means = scaler.mean_.tolist()
scales = scaler.scale_.tolist()

encoder = preprocessor.named_transformers_['cat']
categories_mapping = {}
for col, cats in zip(categorical_cols, encoder.categories_):
    categories_mapping[col] = cats.tolist()

# Fit Ridge model
model = Ridge(alpha=1.0)
model.fit(X_preprocessed, y)

# Align coefficients
encoded_cat_features = list(encoder.get_feature_names_out(categorical_cols))
feature_names = numerical_cols + encoded_cat_features
coefficients = model.coef_.tolist()
intercept = float(model.intercept_)

model_json = {
    "intercept": intercept,
    "numerical_cols": numerical_cols,
    "means": means,
    "scales": scales,
    "categorical_cols": categorical_cols,
    "categories": categories_mapping,
    "feature_names": feature_names,
    "coefficients": coefficients
}

# Write directly to model_data.js
js_content = f"// Automatically generated model coefficients\nconst modelData = {json.dumps(model_json, indent=2)};\n"
with open('model_data.js', 'w') as f:
    f.write(js_content)

print("Export completed. Saved to model_data.js")
