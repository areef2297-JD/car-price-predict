import os
import pickle
import argparse
import pandas as pd
import numpy as np

# Dataset defaults to fill in unprovided features
DEFAULTS = {
    "symboling": 1.0,
    "wheelbase": 97.0,
    "carlength": 173.2,
    "carwidth": 65.5,
    "carheight": 54.1,
    "curbweight": 2422.5,
    "enginesize": 120.0,
    "boreratio": 3.32,
    "stroke": 3.29,
    "compressionratio": 9.0,
    "horsepower": 95.0,
    "peakrpm": 5100.0,
    "citympg": 24.0,
    "highwaympg": 30.0,
    "carbody": "sedan",
    "drivewheel": "fwd",
    "enginelocation": "front",
    "fueltype": "gas",
    "aspiration": "std",
    "doornumber": "four",
    "cylindernumber": "four",
    "enginetype": "ohc",
    "fuelsystem": "mpfi",
    "brand": "toyota"
}

VALID_BRANDS = [
    'alfa-romero', 'audi', 'bmw', 'buick', 'chevrolet', 'dodge', 'honda', 'isuzu', 
    'jaguar', 'mazda', 'mercury', 'mitsubishi', 'nissan', 'peugeot', 'plymouth', 
    'porsche', 'renault', 'saab', 'subaru', 'toyota', 'volkswagen', 'volvo'
]

VALID_BODIES = ['convertible', 'hatchback', 'sedan', 'wagon', 'hardtop']
VALID_DRIVEWHEELS = ['rwd', 'fwd', '4wd']
VALID_FUELS = ['gas', 'diesel']

def load_model():
    model_path = 'car_price_predictor.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'python train.py' first.")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def interactive_mode(model):
    print("=" * 60)
    print("            CAR PRICE PREDICTION INTERACTIVE CLI            ")
    print("=" * 60)
    print("Leave field blank to use standard defaults from the dataset.\n")

    input_data = DEFAULTS.copy()

    # Brand
    print(f"Available brands: {', '.join(VALID_BRANDS[:11])}")
    print(f"                  {', '.join(VALID_BRANDS[11:])}")
    brand = input(f"Enter Car Brand [{DEFAULTS['brand']}]: ").strip().lower()
    if brand:
        if brand in VALID_BRANDS:
            input_data['brand'] = brand
        else:
            print(f"[Warning] Brand '{brand}' not in training vocabulary. Using standard default '{DEFAULTS['brand']}'.")

    # Horsepower
    hp = input(f"Enter Horsepower [{DEFAULTS['horsepower']}]: ").strip()
    if hp:
        try:
            input_data['horsepower'] = float(hp)
        except ValueError:
            print("[Warning] Invalid number. Using default horsepower.")

    # Engine Size
    es = input(f"Enter Engine Size (cc) [{DEFAULTS['enginesize']}]: ").strip()
    if es:
        try:
            input_data['enginesize'] = float(es)
        except ValueError:
            print("[Warning] Invalid number. Using default engine size.")

    # Curb Weight
    cw = input(f"Enter Curb Weight (lbs) [{DEFAULTS['curbweight']}]: ").strip()
    if cw:
        try:
            input_data['curbweight'] = float(cw)
        except ValueError:
            print("[Warning] Invalid number. Using default curb weight.")

    # Car Body
    print(f"Available bodies: {', '.join(VALID_BODIES)}")
    body = input(f"Enter Car Body [{DEFAULTS['carbody']}]: ").strip().lower()
    if body:
        if body in VALID_BODIES:
            input_data['carbody'] = body
        else:
            print(f"[Warning] Body '{body}' invalid. Using default '{DEFAULTS['carbody']}'.")

    # Fuel Type
    fuel = input(f"Enter Fuel Type (gas/diesel) [{DEFAULTS['fueltype']}]: ").strip().lower()
    if fuel:
        if fuel in VALID_FUELS:
            input_data['fueltype'] = fuel
        else:
            print(f"[Warning] Fuel '{fuel}' invalid. Using default '{DEFAULTS['fueltype']}'.")

    # City MPG
    mpg = input(f"Enter City MPG [{DEFAULTS['citympg']}]: ").strip()
    if mpg:
        try:
            input_data['citympg'] = float(mpg)
            # Adjust highway mpg relative to city mpg
            input_data['highwaympg'] = float(mpg) * 1.25
        except ValueError:
            print("[Warning] Invalid number. Using default city MPG.")

    # Predict
    predict_single(model, input_data)

def predict_single(model, input_data):
    df_in = pd.DataFrame([input_data])
    
    # Ensure correct columns order
    columns_order = list(DEFAULTS.keys())
    df_in = df_in[columns_order]
    
    prediction = model.predict(df_in)[0]
    
    print("\n" + "-" * 50)
    print("Input Summary:")
    print(f"  - Brand:         {input_data['brand'].capitalize()}")
    print(f"  - Horsepower:    {input_data['horsepower']} HP")
    print(f"  - Engine Size:   {input_data['enginesize']} cc")
    print(f"  - Curb Weight:   {input_data['curbweight']} lbs")
    print(f"  - Body Style:    {input_data['carbody'].capitalize()}")
    print(f"  - Fuel Type:     {input_data['fueltype'].capitalize()}")
    print(f"  - City MPG:      {input_data['citympg']}")
    print("-" * 50)
    print(f"Predicted Price: ${prediction:,.2f}")
    print("-" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Predict car price using the trained Random Forest model.")
    
    # Individual feature arguments
    parser.add_argument('--brand', type=str, help='Car brand')
    parser.add_argument('--horsepower', type=float, help='Horsepower')
    parser.add_argument('--enginesize', type=float, help='Engine size')
    parser.add_argument('--curbweight', type=float, help='Curb weight')
    parser.add_argument('--carbody', type=str, choices=VALID_BODIES, help='Car body type')
    parser.add_argument('--fueltype', type=str, choices=VALID_FUELS, help='Fuel type')
    parser.add_argument('--citympg', type=float, help='City MPG')
    
    # Batch prediction argument
    parser.add_argument('--csv', type=str, help='Path to a CSV file to make batch predictions')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output CSV path for batch predictions')

    args, unknown = parser.parse_known_args()

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        return

    # Check if we are running in batch CSV mode
    if args.csv:
        if not os.path.exists(args.csv):
            print(f"Error: CSV file '{args.csv}' not found.")
            return
        
        print(f"Reading data from {args.csv}...")
        df_batch = pd.read_csv(args.csv)
        
        # Clean brand in input if it comes from CarName
        if 'CarName' in df_batch.columns and 'brand' not in df_batch.columns:
            df_batch['brand'] = df_batch['CarName'].apply(lambda x: x.split()[0].lower())
            brand_mapping = {
                'maxda': 'mazda', 'toyouta': 'toyota', 'vokswagen': 'volkswagen',
                'vw': 'volkswagen', 'porcshce': 'porsche'
            }
            df_batch['brand'] = df_batch['brand'].replace(brand_mapping)
        
        # Fill missing features with dataset defaults
        for col, default_val in DEFAULTS.items():
            if col not in df_batch.columns:
                df_batch[col] = default_val
            else:
                df_batch[col] = df_batch[col].fillna(default_val)
        
        # Ensure correct column order
        columns_order = list(DEFAULTS.keys())
        X_batch = df_batch[columns_order]
        
        print("Running batch predictions...")
        predictions = model.predict(X_batch)
        
        df_batch['predicted_price'] = predictions
        df_batch.to_csv(args.output, index=False)
        print(f"Successfully saved predictions to {args.output}")
        return

    # If any specific command line arguments are passed, use them
    has_args = any(getattr(args, arg) is not None for arg in ['brand', 'horsepower', 'enginesize', 'curbweight', 'carbody', 'fueltype', 'citympg'])
    
    if has_args:
        input_data = DEFAULTS.copy()
        if args.brand:
            input_data['brand'] = args.brand.lower()
        if args.horsepower:
            input_data['horsepower'] = args.horsepower
        if args.enginesize:
            input_data['enginesize'] = args.enginesize
        if args.curbweight:
            input_data['curbweight'] = args.curbweight
        if args.carbody:
            input_data['carbody'] = args.carbody.lower()
        if args.fueltype:
            input_data['fueltype'] = args.fueltype.lower()
        if args.citympg:
            input_data['citympg'] = args.citympg
            input_data['highwaympg'] = args.citympg * 1.25
        
        predict_single(model, input_data)
    else:
        # Launch Interactive Mode
        interactive_mode(model)

if __name__ == '__main__':
    main()
