import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open("car_price_predictor.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🚗 Car Price Predictor")
st.write("Predict the estimated price of a car using machine learning.")

# -----------------------------
# Input fields
# -----------------------------

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Car Brand",
        [
            "alfa-romero", "audi", "bmw", "buick",
            "chevrolet", "dodge", "honda", "isuzu",
            "jaguar", "mazda", "mercury", "mitsubishi",
            "nissan", "peugeot", "plymouth", "porsche",
            "renault", "saab", "subaru", "toyota",
            "volkswagen", "volvo"
        ]
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=40.0,
        max_value=400.0,
        value=95.0
    )

    enginesize = st.number_input(
        "Engine Size",
        min_value=50.0,
        max_value=500.0,
        value=120.0
    )

    curbweight = st.number_input(
        "Curb Weight",
        min_value=1000.0,
        max_value=5000.0,
        value=2422.5
    )

with col2:
    carbody = st.selectbox(
        "Body Style",
        ["convertible", "hatchback", "sedan", "wagon", "hardtop"]
    )

    fueltype = st.selectbox(
        "Fuel Type",
        ["gas", "diesel"]
    )

    citympg = st.number_input(
        "City MPG",
        min_value=5.0,
        max_value=60.0,
        value=24.0
    )

    drivewheel = st.selectbox(
        "Drive Wheel",
        ["fwd", "rwd", "4wd"]
    )

st.divider()

if st.button("🔮 Predict Car Price", type="primary"):

    input_data = {
        "symboling": 1.0,
        "wheelbase": 97.0,
        "carlength": 173.2,
        "carwidth": 65.5,
        "carheight": 54.1,
        "curbweight": curbweight,
        "enginesize": enginesize,
        "boreratio": 3.32,
        "stroke": 3.29,
        "compressionratio": 9.0,
        "horsepower": horsepower,
        "peakrpm": 5100.0,
        "citympg": citympg,
        "highwaympg": citympg * 1.25,

        "brand": brand,
        "carbody": carbody,
        "drivewheel": drivewheel,
        "enginelocation": "front",
        "fueltype": fueltype,
        "aspiration": "std",
        "doornumber": "four",
        "cylindernumber": "four",
        "enginetype": "ohc",
        "fuelsystem": "mpfi"
    }

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated Car Price: **${prediction:,.2f}**"
    )
