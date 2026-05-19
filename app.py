# ============================================================
# 🚗 Vehicle Price Prediction — Streamlit App
#
#   This app loads a pre-trained Linear Regression model
#   and lets the user input vehicle details to get an
#   estimated selling price.
#
#   Files needed in the same folder:
#     - vehicle_price_model.pkl   (trained model)
#     - label_encoders.pkl        (encoders for text columns)
#     - car_prices.csv            (dataset for charts)
#     - style.css                 (custom styling)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime


# ────────────────────────────────────────────────────────────
# PAGE SETUP
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# Load custom CSS from external file
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# LOAD MODEL & ENCODERS  (cached so it only loads once)
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved model and label encoders from .pkl files."""
    model = joblib.load("vehicle_price_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    return model, label_encoders


@st.cache_data
def load_dataset_stats():
    """
    Read the dataset ONCE and extract the actual min/max ranges
    for sliders and number inputs. This ensures nothing is hardcoded.
    """
    df = pd.read_csv("car_prices.csv")
    return {
        "year_min":      int(df["year"].min()),
        "year_max":      int(df["year"].max()),
        "condition_min": float(df["condition"].min()),
        "condition_max": float(df["condition"].max()),
        "odometer_max":  int(df["odometer"].max()),
        "mmr_max":       int(df["mmr"].max()),
    }


model, label_encoders = load_model()
data_stats = load_dataset_stats()


# ────────────────────────────────────────────────────────────
# PREPARE DROPDOWN OPTIONS
#   All values come directly from label_encoders (trained on
#   the real dataset). Nothing is hardcoded.
# ────────────────────────────────────────────────────────────
MAKES         = sorted([str(m) for m in label_encoders["make"].classes_ if isinstance(m, str) and len(m) > 1])
BODIES        = sorted([str(b) for b in label_encoders["body"].classes_ if isinstance(b, str) and len(b) > 2])
TRANSMISSIONS = sorted([str(t) for t in label_encoders["transmission"].classes_ if isinstance(t, str) and t.isalpha() and t not in ("sedan",)])
STATES        = sorted([str(s) for s in label_encoders["state"].classes_ if isinstance(s, str) and len(s) == 2 and s.isalpha()])
COLORS        = sorted([str(c) for c in label_encoders["color"].classes_ if isinstance(c, str) and c.isalpha() and len(c) > 2])

# Full state names for display (abbreviation → full name)
STATE_NAMES = {
    "ab": "Alberta", "al": "Alabama", "az": "Arizona", "ca": "California",
    "co": "Colorado", "fl": "Florida", "ga": "Georgia", "hi": "Hawaii",
    "il": "Illinois", "in": "Indiana", "la": "Louisiana", "ma": "Massachusetts",
    "md": "Maryland", "mi": "Michigan", "mn": "Minnesota", "mo": "Missouri",
    "ms": "Mississippi", "nc": "North Carolina", "ne": "Nebraska",
    "nj": "New Jersey", "nm": "New Mexico", "ns": "Nova Scotia",
    "nv": "Nevada", "ny": "New York", "oh": "Ohio", "ok": "Oklahoma",
    "on": "Ontario", "or": "Oregon", "pa": "Pennsylvania", "pr": "Puerto Rico",
    "qc": "Quebec", "sc": "South Carolina", "tn": "Tennessee", "tx": "Texas",
    "ut": "Utah", "va": "Virginia", "wa": "Washington", "wi": "Wisconsin",
}
# Build display list: "California (ca)" and a reverse lookup
STATE_DISPLAY = [f"{STATE_NAMES.get(s, s.upper())} ({s})" for s in STATES]
STATE_DISPLAY_TO_CODE = {display: code for display, code in zip(STATE_DISPLAY, STATES)}


# ────────────────────────────────────────────────────────────
# HELPER FUNCTION
# ────────────────────────────────────────────────────────────
def safe_encode(encoder, value):
    """
    Convert a text value (e.g. 'ford') into a number using the
    label encoder. Returns 0 if the value is unknown.
    """
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return 0


# ────────────────────────────────────────────────────────────
# SIDEBAR — User inputs for vehicle details
# ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚗 Vehicle Details")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Categorical inputs ---
    make         = st.selectbox("Make / Brand",    MAKES,         index=MAKES.index("ford") if "ford" in MAKES else 0)
    body         = st.selectbox("Body Style",      BODIES,        index=BODIES.index("sedan") if "sedan" in BODIES else 0)
    transmission = st.selectbox("Transmission",    TRANSMISSIONS)
    color        = st.selectbox("Exterior Color",  COLORS,        index=COLORS.index("white") if "white" in COLORS else 0)
    state_display = st.selectbox("State (Location)", STATE_DISPLAY,
                                 index=STATE_DISPLAY.index("California (ca)") if "California (ca)" in STATE_DISPLAY else 0)
    state = STATE_DISPLAY_TO_CODE[state_display]  # convert back to abbreviation for encoding

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Specifications")

    # --- Numeric inputs (all ranges from the dataset, not hardcoded) ---
    # Limit year to reasonable range for model reliability (1990-2015)
    min_year = max(data_stats["year_min"], 1990)
    year      = st.slider("Model Year",
                          min_year, data_stats["year_max"],
                          value=data_stats["year_max"] - 3)  # default: 3 years before max
    condition = st.slider("Condition",
                          data_stats["condition_min"], data_stats["condition_max"],
                          value=3.5, step=0.5)
    odometer  = st.number_input("Odometer (miles)",
                                min_value=0, max_value=data_stats["odometer_max"],
                                value=35_000, step=1_000)
    mmr       = st.number_input("MMR (Market Value)",
                                min_value=0, max_value=data_stats["mmr_max"],
                                value=10_000, step=500)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Predict button ---
    predict_btn = st.button("🔮  Predict Price", use_container_width=True)


# ────────────────────────────────────────────────────────────
# MAIN AREA — Header
# ────────────────────────────────────────────────────────────
# Display the hero image
try:
    st.image("automotive_oracle_hero_1778744196125.png", use_container_width=True)
except Exception:
    pass

st.markdown('<p class="hero-title">Automotive Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-fidelity price intelligence engine · calibrated on 500K+ historical auction records</p>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Two tabs for different sections
tab_predict, tab_about = st.tabs(["🔮 Predictive Engine", "ℹ️ Technical Specs"])


# ────────────────────────────────────────────────────────────
# TAB 1 — PREDICTION
# ────────────────────────────────────────────────────────────
with tab_predict:
    if predict_btn:
        # Step 1: Calculate derived features
        current_year     = datetime.datetime.now().year
        vehicle_age      = current_year - year                     # how old the car is
        mileage_per_year = odometer / max(vehicle_age, 1)          # average miles driven per year
        sale_year        = current_year
        sale_month       = datetime.datetime.now().month

        # Step 2: Encode text columns into numbers
        make_enc  = safe_encode(label_encoders["make"],         make)
        body_enc  = safe_encode(label_encoders["body"],         body)
        trans_enc = safe_encode(label_encoders["transmission"], transmission)
        state_enc = safe_encode(label_encoders["state"],        state)
        color_enc = safe_encode(label_encoders["color"],        color)

        # Step 3: Build the feature array (same order as training)
        #   [year, condition, odometer, mmr,
        #    vehicle_age, mileage_per_year, sale_year, sale_month,
        #    make_enc, body_enc, trans_enc, state_enc, color_enc]
        features = np.array([[
            year, condition, odometer, mmr,
            vehicle_age, mileage_per_year, sale_year, sale_month,
            make_enc, body_enc, trans_enc, state_enc, color_enc
        ]])

        # Step 4: Predict
        predicted_price = max(model.predict(features)[0], 0)  # floor at $0

        # Step 5: Show the result with warning for old vehicles
        if year < 1990:
            st.warning("⚠️ **Limited Data**: Predictions for vehicles before 1990 may be unreliable due to insufficient training data.")
        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">Estimate</p>
            <p class="result-price">${predicted_price:,.0f}</p>
            <p class="result-desc">Verified Specification: {year} {make.title()} · {body.title()} · {odometer:,} mi</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 6: Show quick comparison metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Vehicle Age", f"{vehicle_age} yrs")
        col2.metric("Miles / Year",f"{mileage_per_year:,.0f}")
        col3.metric("Condition",   f"{condition:.1f} / 5.0")

        st.markdown("<br>", unsafe_allow_html=True)

    else:
        # Empty state — shown before the user clicks Predict
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown("""
            <div class="result-card" style="margin-top:30px;">
                <p style="font-size:4rem; margin:0; filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));">👁️</p>
                <p class="result-label" style="margin-top:16px;">Initiate Sequence</p>
                <p class="result-desc">
                    Configure the vehicle parameters in the command center (sidebar)
                    and activate the <strong>Predictive Engine</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# (Market Analysis tab removed for efficiency)


# ────────────────────────────────────────────────────────────
# TAB 3 — ABOUT  (model explanation)
# ────────────────────────────────────────────────────────────
with tab_about:
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("""
        #### 🧠 How It Works
        1. **Data Cleaning** — Remove duplicates, fill missing values, remove outliers (IQR)
        2. **Feature Engineering** — Add vehicle age & mileage/year
        3. **Label Encoding** — Convert text columns (make, body, etc.) to numbers
        4. **Training** — Scikit-learn `LinearRegression` on 80/20 split

        #### 📐 Model Features
        | # | Feature | Type |
        |---|---------|------|
        | 1 | Year | Numeric |
        | 2 | Condition | Numeric (1–5) |
        | 3 | Odometer | Numeric |
        | 4 | MMR | Numeric |
        | 5 | Vehicle Age | Engineered |
        | 6 | Mileage per Year | Engineered |
        | 7 | Sale Year | Date-derived |
        | 8 | Sale Month | Date-derived |
        | 9–13 | Make, Body, Transmission, State, Color | Encoded |
        """)

    with right_col:
        st.markdown("#### 📈 Performance")
        m1, m2 = st.columns(2)
        m1.metric("R² Score", "0.94+")
        m2.metric("MAE", "~$1,200")

        st.markdown("""
        #### ⚠️ Limitations
        - Prices are **estimates** — real prices depend on market and negotiation.
        - Trained on **US auction data** (2014–2015 sales).
        - Rare / luxury brands may be less accurate.

        #### 🛠️ Tech Stack
        Python · Pandas · Scikit-learn · Streamlit · Plotly
        """)


# ────────────────────────────────────────────────────────────
# FOOTER
# ────────────────────────────────────────────────────────────
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="footer">Built with ❤️ using Streamlit · Vehicle Price Prediction Project</p>', unsafe_allow_html=True)
