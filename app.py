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
import plotly.express as px
import plotly.graph_objects as go


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


model, label_encoders = load_model()


# ────────────────────────────────────────────────────────────
# PREPARE DROPDOWN OPTIONS
#   We pull the valid values from label_encoders so dropdowns
#   only show values the model was trained on.
# ────────────────────────────────────────────────────────────
MAKES = sorted([m for m in label_encoders["make"].classes_ if len(m) > 1])
BODIES = sorted([b for b in label_encoders["body"].classes_ if len(b) > 2])
TRANSMISSIONS = ["automatic", "manual"]
STATES = sorted([s for s in label_encoders["state"].classes_ if len(s) == 2 and s.isalpha()])
COLORS = sorted([c for c in label_encoders["color"].classes_ if c.isalpha() and len(c) > 2])


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
    state        = st.selectbox("State (Location)",STATES,        index=STATES.index("ca") if "ca" in STATES else 0)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Specifications")

    # --- Numeric inputs ---
    # Year range 1982-2015 matches the training dataset
    year      = st.slider("Model Year",      1982, 2015, 2012)
    condition = st.slider("Condition (1–5)",  1.0, 5.0, 3.5, 0.5)
    odometer  = st.number_input("Odometer (miles)",    min_value=0, max_value=500_000, value=35_000, step=1_000)
    mmr       = st.number_input("MMR (Market Value $)", min_value=0, max_value=250_000, value=18_000, step=500)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Predict button ---
    predict_btn = st.button("🔮  Predict Price", use_container_width=True)


# ────────────────────────────────────────────────────────────
# MAIN AREA — Header
# ────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🚗 Vehicle Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">AI-powered pricing using Linear Regression · trained on 500K+ auction records</p>', unsafe_allow_html=True)
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# Three tabs for different sections
tab_predict, tab_explore, tab_about = st.tabs(["🔮 Prediction", "📊 Explore Data", "ℹ️ About"])


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

        # Step 5: Show the result
        st.markdown(f"""
        <div class="result-card">
            <p class="result-label">Estimated Selling Price</p>
            <p class="result-price">${predicted_price:,.0f}</p>
            <p class="result-desc">{year} {make.title()} · {body.title()} · {odometer:,} mi</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 6: Show quick comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        diff = predicted_price - mmr
        col1.metric("vs MMR",      f"${diff:+,.0f}", delta=f"{diff/max(mmr,1)*100:+.1f}%")
        col2.metric("Vehicle Age", f"{vehicle_age} yrs")
        col3.metric("Miles / Year",f"{mileage_per_year:,.0f}")
        col4.metric("Condition",   f"{condition:.1f} / 5.0")

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 7: Feature contribution chart
        #   Shows how much each feature pushes the price up or down.
        #   contribution = model_coefficient × feature_value
        st.markdown("#### 🧩 Feature Contributions")
        feature_names = [
            "Year", "Condition", "Odometer", "MMR",
            "Vehicle Age", "Miles/Year", "Sale Year", "Sale Month",
            "Make", "Body", "Transmission", "State", "Color"
        ]
        contributions = model.coef_ * features[0]
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution ($)": contributions
        }).sort_values("Contribution ($)", ascending=True)

        fig = px.bar(
            contrib_df, x="Contribution ($)", y="Feature",
            orientation="h",
            color="Contribution ($)",
            color_continuous_scale=["#ec4899", "#6366f1", "#06b6d4"],
            template="plotly_dark"
        )
        fig.update_layout(
            height=420, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"), yaxis_title="", margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Empty state — shown before the user clicks Predict
        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.markdown("""
            <div class="result-card" style="margin-top:30px;">
                <p style="font-size:3rem; margin:0;">🚗</p>
                <p class="result-label" style="margin-top:12px;">Get Started</p>
                <p class="result-desc">
                    Fill in the vehicle details in the sidebar and hit
                    <strong>Predict Price</strong>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# TAB 2 — EXPLORE DATA  (interactive charts from the dataset)
# ────────────────────────────────────────────────────────────
with tab_explore:
    st.markdown("#### 📊 Dataset Insights")

    # Load a 50K sample for fast chart rendering
    @st.cache_data
    def get_sample():
        df = pd.read_csv("car_prices.csv")
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
        return df.sample(min(50_000, len(df)), random_state=42)

    sample = get_sample()

    # --- Row 1: Price histogram + Price vs Odometer scatter ---
    left, right = st.columns(2)

    with left:
        st.markdown("##### Price Distribution")
        fig1 = px.histogram(sample, x="sellingprice", nbins=80,
                            color_discrete_sequence=["#6366f1"], template="plotly_dark")
        fig1.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"), margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.markdown("##### Price vs Odometer")
        scatter_sample = sample.sample(min(5000, len(sample)), random_state=1)
        fig2 = px.scatter(scatter_sample, x="odometer", y="sellingprice", color="year",
                          color_continuous_scale="Plasma", opacity=0.55, template="plotly_dark")
        fig2.update_traces(marker_size=4)
        fig2.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"), margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Row 2: Top makes bar + Avg price by year line ---
    left2, right2 = st.columns(2)

    with left2:
        st.markdown("##### Top 15 Makes by Avg Price")
        top_makes = (sample.groupby("make")["sellingprice"]
                     .mean().sort_values(ascending=False).head(15).reset_index())
        fig3 = px.bar(top_makes, x="sellingprice", y="make", orientation="h",
                      color="sellingprice", color_continuous_scale=["#6366f1","#a855f7","#ec4899"],
                      template="plotly_dark")
        fig3.update_layout(height=420, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"), yaxis=dict(autorange="reversed"),
                           margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with right2:
        st.markdown("##### Avg Price by Year")
        year_avg = sample.groupby("year")["sellingprice"].mean().reset_index().sort_values("year")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=year_avg["year"], y=year_avg["sellingprice"],
            mode="lines+markers",
            line=dict(color="#a855f7", width=3),
            marker=dict(size=6, color="#ec4899"),
            fill="tozeroy", fillcolor="rgba(168,85,247,0.10)",
        ))
        fig4.update_layout(height=420, template="plotly_dark",
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"), xaxis_title="Year", yaxis_title="Avg Price ($)",
                           margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # --- Row 3: Correlation heatmap ---
    st.markdown("##### Correlation Heatmap")
    num_cols = [c for c in ["year","condition","odometer","mmr","sellingprice"] if c in sample.columns]
    fig5 = px.imshow(sample[num_cols].corr(), text_auto=".2f",
                     color_continuous_scale=["#1e1b4b","#6366f1","#ec4899"], template="plotly_dark")
    fig5.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(family="Inter"), margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig5, use_container_width=True)


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
        | 4 | MMR (Market Value) | Numeric |
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
