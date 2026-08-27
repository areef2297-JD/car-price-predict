import os
import pickle
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        /* Main background */
        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(0, 242, 254, 0.08),
                    transparent 30%
                ),
                radial-gradient(
                    circle at 90% 20%,
                    rgba(155, 81, 224, 0.08),
                    transparent 30%
                ),
                #080b14;
            color: #f5f7ff;
        }

        /* Main container */
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        /* Header */
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            background: linear-gradient(
                90deg,
                #00f2fe,
                #9b51e0
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            text-align: center;
            color: #a7adbd;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        /* Cards */
        .glass-card {
            background: rgba(18, 23, 38, 0.85);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow:
                0 10px 40px rgba(0,0,0,0.25),
                inset 0 1px 0 rgba(255,255,255,0.03);
        }

        /* Price card */
        .price-card {
            background:
                linear-gradient(
                    135deg,
                    rgba(0,242,254,0.12),
                    rgba(155,81,224,0.15)
                );
            border: 1px solid rgba(0,242,254,0.25);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 1rem;
        }

        .price-label {
            color: #a7adbd;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .price-value {
            font-size: 3rem;
            font-weight: 800;
            color: #00f2fe;
            margin: 0.5rem 0;
        }

        .price-note {
            color: #8f96a8;
            font-size: 0.85rem;
        }

        /* Contribution rows */
        .contribution-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }

        .contribution-row:last-child {
            border-bottom: none;
        }

        .contribution-label {
            color: #d8dcea;
        }

        .positive {
            color: #00e676;
            font-weight: 700;
        }

        .negative {
            color: #ff5252;
            font-weight: 700;
        }

        .neutral {
            color: #a7adbd;
            font-weight: 700;
        }

        /* Section headings */
        .section-title {
            color: #ffffff;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        /* Streamlit tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 20px;
            border-radius: 10px;
            color: #a7adbd;
        }

        .stTabs [aria-selected="true"] {
            color: #00f2fe !important;
            background: rgba(0,242,254,0.08);
        }

        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(0,242,254,0.4);
            background: linear-gradient(
                90deg,
                rgba(0,242,254,0.15),
                rgba(155,81,224,0.15)
            );
            color: white;
            font-weight: 700;
        }

        .stButton > button:hover {
            border-color: #00f2fe;
            color: #00f2fe;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Footer */
        .footer {
            text-align: center;
            color: #73798a;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.06);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_path = "car_price_predictor.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found."
        )

    with open(model_path, "rb") as file:
        return pickle.load(file)


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.exception(e)
    st.stop()


# ============================================================
# DEFAULTS
# These match your existing predict.py / index.html
# ============================================================

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

# These are the categories used by your existing model_data.js.
BRANDS = [
    "alfa-romero",
    "audi",
    "bmw",
    "buick",
    "chevrolet",
    "dodge",
    "honda",
    "isuzu",
    "jaguar",
    "mazda",
    "mercury",
    "mitsubishi",
    "nissan",
    "peugeot",
    "plymouth",
    "porsche",
    "renault",
    "saab",
    "subaru",
    "toyota",
    "volkswagen",
    "volvo"
]

BODY_TYPES = [
    "convertible",
    "hardtop",
    "hatchback",
    "sedan",
    "wagon"
]

DRIVE_WHEELS = [
    "4wd",
    "fwd",
    "rwd"
]

ENGINE_LOCATIONS = [
    "front",
    "rear"
]

FUEL_TYPES = [
    "diesel",
    "gas"
]

ASPIRATIONS = [
    "std",
    "turbo"
]

DOORS = [
    "four",
    "two"
]

CYLINDERS = [
    "eight",
    "five",
    "four",
    "six",
    "three",
    "twelve",
    "two"
]

ENGINE_TYPES = [
    "dohc",
    "dohcv",
    "l",
    "ohc",
    "ohcf",
    "ohcv",
    "rotor"
]

FUEL_SYSTEMS = [
    "1bbl",
    "2bbl",
    "4bbl",
    "idi",
    "mfi",
    "mpfi",
    "spdi",
    "spfi"
]


def pretty_name(value):
    """Make values like alfa-romero look nicer."""
    return str(value).replace("-", " ").title()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚗 Car Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-time machine learning price predictions powered by your trained model'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MAIN TABS
# ============================================================

main_predictor, training_report = st.tabs(
    [
        "🔮 Interactive Predictor",
        "📊 Model Training Report"
    ]
)


# ============================================================
# PREDICTOR
# ============================================================

with main_predictor:

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    engine_tab, body_tab, dimensions_tab = st.tabs(
        [
            "⚙️ Engine & Power",
            "🚘 Body & Style",
            "📐 Dimensions & Fuel"
        ]
    )

    # --------------------------------------------------------
    # ENGINE & POWER
    # --------------------------------------------------------

    with engine_tab:

        col1, col2 = st.columns(2)

        with col1:
            horsepower = st.slider(
                "Horsepower",
                min_value=48,
                max_value=288,
                value=95,
                step=1
            )

            enginesize = st.slider(
                "Engine Size (cc)",
                min_value=61,
                max_value=326,
                value=120,
                step=1
            )

            peakrpm = st.slider(
                "Peak RPM",
                min_value=4150,
                max_value=6600,
                value=5100,
                step=50
            )

            fueltype = st.selectbox(
                "Fuel Type",
                FUEL_TYPES,
                index=FUEL_TYPES.index(DEFAULTS["fueltype"]),
                format_func=pretty_name
            )

        with col2:
            aspiration = st.selectbox(
                "Aspiration",
                ASPIRATIONS,
                index=ASPIRATIONS.index(DEFAULTS["aspiration"]),
                format_func=pretty_name
            )

            cylindernumber = st.selectbox(
                "Cylinders",
                CYLINDERS,
                index=CYLINDERS.index(DEFAULTS["cylindernumber"]),
                format_func=pretty_name
            )

            enginetype = st.selectbox(
                "Engine Type",
                ENGINE_TYPES,
                index=ENGINE_TYPES.index(DEFAULTS["enginetype"]),
                format_func=pretty_name
            )

    # --------------------------------------------------------
    # BODY & STYLE
    # --------------------------------------------------------

    with body_tab:

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                "Brand",
                BRANDS,
                index=BRANDS.index(DEFAULTS["brand"]),
                format_func=pretty_name
            )

            carbody = st.selectbox(
                "Body Style",
                BODY_TYPES,
                index=BODY_TYPES.index(DEFAULTS["carbody"]),
                format_func=pretty_name
            )

            drivewheel = st.selectbox(
                "Drive Wheel",
                DRIVE_WHEELS,
                index=DRIVE_WHEELS.index(DEFAULTS["drivewheel"]),
                format_func=pretty_name
            )

        with col2:
            enginelocation = st.selectbox(
                "Engine Location",
                ENGINE_LOCATIONS,
                index=ENGINE_LOCATIONS.index(DEFAULTS["enginelocation"]),
                format_func=pretty_name
            )

            doornumber = st.selectbox(
                "Doors",
                DOORS,
                index=DOORS.index(DEFAULTS["doornumber"]),
                format_func=pretty_name
            )

    # --------------------------------------------------------
    # DIMENSIONS & FUEL
    # --------------------------------------------------------

    with dimensions_tab:

        col1, col2 = st.columns(2)

        with col1:
            curbweight = st.slider(
                "Curb Weight (lbs)",
                min_value=1488.0,
                max_value=4066.0,
                value=2422.5,
                step=5.0
            )

            carlength = st.slider(
                "Car Length (in)",
                min_value=141.1,
                max_value=208.1,
                value=173.2,
                step=0.1
            )

            carwidth = st.slider(
                "Car Width (in)",
                min_value=60.3,
                max_value=72.3,
                value=65.5,
                step=0.1
            )

            carheight = st.slider(
                "Car Height (in)",
                min_value=47.8,
                max_value=59.8,
                value=54.1,
                step=0.1
            )

            wheelbase = st.slider(
                "Wheelbase (in)",
                min_value=86.6,
                max_value=120.9,
                value=97.0,
                step=0.1
            )

        with col2:
            compressionratio = st.slider(
                "Compression Ratio",
                min_value=7.0,
                max_value=23.0,
                value=9.0,
                step=0.1
            )

            boreratio = st.slider(
                "Bore Ratio",
                min_value=2.54,
                max_value=3.94,
                value=3.32,
                step=0.01
            )

            stroke = st.slider(
                "Stroke",
                min_value=2.07,
                max_value=4.17,
                value=3.29,
                step=0.01
            )

            citympg = st.slider(
                "City MPG",
                min_value=13,
                max_value=49,
                value=24,
                step=1
            )

            highwaympg = round(citympg * 1.25, 1)

            st.caption(
                f"Highway MPG is automatically calculated as "
                f"City MPG × 1.25 → **{highwaympg} MPG**"
            )

            symboling = st.slider(
                "Symboling Rating",
                min_value=-3,
                max_value=3,
                value=1,
                step=1
            )

            fuelsystem = st.selectbox(
                "Fuel System",
                FUEL_SYSTEMS,
                index=FUEL_SYSTEMS.index(DEFAULTS["fuelsystem"]),
                format_func=pretty_name
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # PREDICTION
    # ========================================================

    input_data = {
        "symboling": float(symboling),
        "wheelbase": float(wheelbase),
        "carlength": float(carlength),
        "carwidth": float(carwidth),
        "carheight": float(carheight),
        "curbweight": float(curbweight),
        "enginesize": float(enginesize),
        "boreratio": float(boreratio),
        "stroke": float(stroke),
        "compressionratio": float(compressionratio),
        "horsepower": float(horsepower),
        "peakrpm": float(peakrpm),
        "citympg": float(citympg),
        "highwaympg": float(highwaympg),
        "carbody": carbody,
        "drivewheel": drivewheel,
        "enginelocation": enginelocation,
        "fueltype": fueltype,
        "aspiration": aspiration,
        "doornumber": doornumber,
        "cylindernumber": cylindernumber,
        "enginetype": enginetype,
        "fuelsystem": fuelsystem,
        "brand": brand
    }

    # Keep exact feature order from predict.py
    feature_order = list(DEFAULTS.keys())

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_order
    )

    # Predict
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    # ========================================================
    # RESULTS
    # ========================================================

    result_col, contribution_col = st.columns(
        [1, 1],
        gap="large"
    )

    with result_col:

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Value</div>
                <div class="price-value">
                    ${prediction:,.2f}
                </div>
                <div class="price-note">
                    Calculated using your trained machine-learning model
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🚘 Input Summary</div>',
            unsafe_allow_html=True
        )

        summary_data = {
            "Feature": [
                "Brand",
                "Body Style",
                "Horsepower",
                "Engine Size",
                "Curb Weight",
                "Fuel Type",
                "City MPG",
                "Highway MPG"
            ],
            "Value": [
                pretty_name(brand),
                pretty_name(carbody),
                f"{horsepower} HP",
                f"{enginesize} cc",
                f"{curbweight:,.1f} lbs",
                pretty_name(fueltype),
                f"{citympg} MPG",
                f"{highwaympg} MPG"
            ]
        }

        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # FEATURE CONTRIBUTIONS
    # ========================================================

    with contribution_col:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">📈 Feature Contributions</div>',
            unsafe_allow_html=True
        )

        st.info(
            "The prediction shown above comes directly from the trained "
            "Python model. The contribution values below are grouped "
            "approximations for interpretability."
        )

        # A simple grouped sensitivity/importance view.
        #
        # We calculate the effect of changing one numerical feature
        # while keeping everything else constant.
        base_prediction = prediction

        groups = {
            "Engine & Power": [
                "horsepower",
                "enginesize",
                "peakrpm"
            ],
            "Dimensions": [
                "curbweight",
                "carlength",
                "carwidth",
                "carheight",
                "wheelbase",
                "compressionratio",
                "boreratio",
                "stroke"
            ],
            "Fuel Efficiency": [
                "citympg",
                "highwaympg"
            ],
            "Other Specifications": [
                "symboling"
            ]
        }

        contributions = {}

        for group_name, features in groups.items():

            total_effect = 0.0

            for feature in features:

                modified = input_data.copy()

                # Move numerical feature toward its dataset default.
                modified[feature] = DEFAULTS[feature]

                modified_df = pd.DataFrame(
                    [modified],
                    columns=feature_order
                )

                try:
                    modified_prediction = float(
                        model.predict(modified_df)[0]
                    )

                    total_effect += base_prediction - modified_prediction

                except Exception:
                    pass

            contributions[group_name] = total_effect

        # Brand effect
        brand_modified = input_data.copy()
        brand_modified["brand"] = DEFAULTS["brand"]

        try:
            brand_prediction = float(
                model.predict(
                    pd.DataFrame(
                        [brand_modified],
                        columns=feature_order
                    )
                )[0]
            )

            contributions["Brand"] = (
                base_prediction - brand_prediction
            )

        except Exception:
            contributions["Brand"] = 0.0

        # Render contribution rows
        for label, value in contributions.items():

            if value > 50:
                css_class = "positive"
                prefix = "+"

            elif value < -50:
                css_class = "negative"
                prefix = "-"

            else:
                css_class = "neutral"
                prefix = ""

            st.markdown(
                f"""
                <div class="contribution-row">
                    <span class="contribution-label">
                        {label}
                    </span>
                    <span class="{css_class}">
                        {prefix}${abs(value):,.2f}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODEL TRAINING REPORT
# ============================================================

with training_report:

    st.markdown(
        '<div class="glass-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">📊 Model Evaluation Report</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        This dashboard shows the evaluation results of the three
        regression algorithms used in your project.
        """
    )

    evaluation_data = pd.DataFrame(
        {
            "Algorithm": [
                "Random Forest",
                "Ridge Regression",
                "Gradient Boosting"
            ],
            "Test R²": [
                0.9183,
                0.8872,
                0.9125
            ],
            "Test MAE": [
                1841.23,
                2016.00,
                1861.64
            ],
            "Test RMSE": [
                2480.82,
                2915.87,
                2566.87
            ]
        }
    )

    st.dataframe(
        evaluation_data.style.format(
            {
                "Test R²": "{:.4f}",
                "Test MAE": "${:,.2f}",
                "Test RMSE": "${:,.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # METRIC CARDS
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🏆 Best R²",
            "0.9183",
            "Random Forest"
        )

    with col2:
        st.metric(
            "💰 Best MAE",
            "$1,841.23",
            "Random Forest"
        )

    with col3:
        st.metric(
            "📉 Best RMSE",
            "$2,480.82",
            "Random Forest"
        )

    st.markdown("---")

    # ========================================================
    # DIAGNOSTIC VISUALIZATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Diagnostic Visualizations</div>',
        unsafe_allow_html=True
    )

    plots = [
        (
            "Actual Price vs. Predicted Price",
            "plots/actual_vs_predicted.png"
        ),
        (
            "Residuals Analysis & Errors Distribution",
            "plots/residuals_analysis.png"
        ),
        (
            "Top 15 Feature Importances",
            "plots/feature_importance.png"
        )
    ]

    plot_columns = st.columns(2)

    for index, (title, path) in enumerate(plots):

        with plot_columns[index % 2]:

            st.markdown(
                '<div class="glass-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                f"### {title}"
            )

            if os.path.exists(path):
                st.image(
                    path,
                    use_container_width=True
                )
            else:
                st.warning(
                    f"Plot not found: `{path}`"
                )

            st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🚗 Car Price Predictor |
        Machine Learning Regression Project
    </div>
    """,
    unsafe_allow_html=True
)
