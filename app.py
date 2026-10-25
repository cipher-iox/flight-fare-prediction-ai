"""
SkyFare — Commercial Flight Price Estimator & Market Analytics
AIC354 Machine Learning Fundamentals | Assignment 3
Production Deployment: Streamlit Community Cloud
"""

import os
import pickle
import warnings
from datetime import datetime, date

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyFare | Domestic Flight Price Index",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clean Commercial Airline Styling (Sandy & Black Matte Theme) ─────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; 
}

/* Background - Matte Obsidian Black */
.stApp {
    background-color: #121212;
    color: #ede5d8;
}

#MainMenu, footer, header { visibility: hidden; }

/* Global typography */
h1, h2, h3, h4, h5, h6 {
    color: #f5efe6 !important;
}
p, span, label {
    color: #dfcfbe !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #171615 !important;
    border-right: 1px solid #2b2823 !important;
}
section[data-testid="stSidebar"] * {
    color: #dfcfbe;
}

/* Form inputs (Select, Date, Number) */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
input {
    background-color: #1c1b18 !important;
    color: #f5efe6 !important;
    border-color: #3d3830 !important;
}

/* Header Banner - Rich Matte Black with Warm Sand Accent */
.portal-header {
    background: #191816;
    border: 1px solid #332f28;
    border-radius: 8px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.portal-badge {
    display: inline-block;
    background: #26231e;
    color: #cbb592 !important;
    border: 1px solid #4a4235;
    padding: 3px 10px;
    border-radius: 4px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.portal-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f5efe6 !important;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.portal-desc {
    color: #a89d8f !important;
    font-size: 0.94rem;
    margin-top: 0.5rem;
    max-width: 720px;
}

/* KPI Summary Cards */
.kpi-box {
    background: #191816;
    border: 1px solid #332f28;
    border-radius: 8px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: border-color 0.2s ease;
}
.kpi-box:hover {
    border-color: #8c7b66;
}
.kpi-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #cbb592 !important;
    letter-spacing: -0.5px;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #8f8375 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-top: 4px;
}

/* Flight Ticket Card Aesthetic */
.ticket-card {
    background: #191816;
    border: 1px solid #332f28;
    border-top: 3px solid #cbb592;
    border-radius: 8px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
}
.ticket-route {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed #36322a;
    padding-bottom: 1.2rem;
    margin-bottom: 1.2rem;
}
.route-city {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f5efe6 !important;
}
.route-airline {
    font-size: 0.82rem;
    color: #8f8375 !important;
    font-weight: 600;
}
.flight-icon-center {
    font-size: 1.3rem;
    color: #cbb592 !important;
}
.ticket-price-val {
    font-size: 3.2rem;
    font-weight: 800;
    color: #f5efe6 !important;
    line-height: 1;
    margin: 0.4rem 0 0.2rem;
    letter-spacing: -1px;
}
.ticket-price-sub {
    font-size: 0.8rem;
    color: #8f8375 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-weight: 600;
}

/* Deal Badge */
.deal-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    margin-top: 0.8rem;
}

.pill-card {
    background: #151412;
    border: 1px solid #2d2a23;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    text-align: center;
}
.pill-card-val {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f5efe6 !important;
}
.pill-card-lbl {
    font-size: 0.7rem;
    color: #8f8375 !important;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 2px;
}

/* Corporate Buttons - Solid Sand & Black */
div.stButton > button:first-child {
    background-color: #cbb592 !important;
    color: #121212 !important;
    border: 1px solid #cbb592 !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
div.stButton > button:first-child:hover {
    background-color: #dfcfbe !important;
    border-color: #dfcfbe !important;
    color: #121212 !important;
}

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #161513;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid #2d2a23;
}
.stTabs [data-baseweb="tab"] {
    height: 38px;
    border-radius: 6px;
    color: #8f8375 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0 16px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #262420 !important;
    color: #f5efe6 !important;
    border: 1px solid #4a4235 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Airline & City Options ────────────────────────────────────────────────────
AIRLINES = [
    'IndiGo',
    'Air India',
    'Jet Airways',
    'SpiceJet',
    'Multiple carriers',
    'GoAir',
    'Vistara',
    'Air Asia',
    'Multiple carriers Premium economy',
    'Jet Airways Business',
    'Vistara Premium economy',
    'Trujet'
]

SOURCES = ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai']
DESTINATIONS = ['Cochin', 'Banglore', 'Delhi', 'Hyderabad', 'Kolkata']

STOPS_MAP = {
    'Non-stop (Direct)': 0,
    '1 Stop': 1,
    '2 Stops': 2,
    '3 Stops': 3,
    '4 Stops': 4
}

# ── Load Model Bundle ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model_bundle = load_model()

# ── Load Dataset for EDA ───────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset', 'Data_Train.csv')
    xlsx_path = os.path.join(os.path.dirname(__file__), 'dataset', 'Data_Train.xlsx')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    return None

df = load_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
      <div style="font-size:1.5rem; color:#cbb592;">✈</div>
      <div>
        <div style="font-weight:800; font-size:1.15rem; color:#f5efe6;">SkyFare</div>
        <div style="font-size:0.75rem; color:#8f8375;">Flight Price Index</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model_bundle:
        st.markdown("""
        <div style="background:#201e19; border:1px solid #3d382f; border-radius:6px; padding:6px 10px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#cbb592;"></span>
          <span style="font-size:0.8rem; color:#cbb592; font-weight:600;">Pricing Service Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#261817; border:1px solid #4a2824; border-radius:6px; padding:6px 10px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#dfa497;"></span>
          <span style="font-size:0.8rem; color:#dfa497; font-weight:600;">Pricing Service Offline</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Currency Settings")
    currency = st.selectbox("Display Currency", [
        "INR (₹) — Indian Rupee",
        "USD ($) — US Dollar",
        "EUR (€) — Euro",
        "GBP (£) — British Pound",
        "AED (AED) — UAE Dirham",
        "PKR (Rs.) — Pakistani Rupee"
    ], index=0)

    # Conversion rates relative to INR
    curr_rates = {
        "INR (₹) — Indian Rupee": (1.0, "₹"),
        "USD ($) — US Dollar": (1 / 83.5, "$"),
        "EUR (€) — Euro": (1 / 90.2, "€"),
        "GBP (£) — British Pound": (1 / 105.4, "£"),
        "AED (AED) — UAE Dirham": (1 / 22.7, "AED "),
        "PKR (Rs.) — Pakistani Rupee": (3.32, "Rs. ")
    }
    fx_rate, curr_sym = curr_rates[currency]

    st.markdown("---")
    st.markdown("#### Database Summary")
    st.markdown("""
    - **Itinerary Records:** 10,683 Commercial Flights
    - **Validation Protocol:** 10-Fold Cross-Validation
    - **Standard Currency:** Indian Rupee (INR ₹)
    """)

    if model_bundle:
        st.markdown("---")
        st.markdown("#### Reference Model")
        st.markdown(f"**Selected Model:** `{model_bundle.get('model_name', 'Random Forest Regressor')}`")
        if 'rmse' in model_bundle:
            st.markdown(f"**Validation RMSE:** `₹{model_bundle['rmse']:,.2f}`")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER / BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="portal-header">
  <div class="portal-badge">Commercial Aviation Pricing</div>
  <h1 class="portal-title">Domestic Flight Fare Search & Rate Index</h1>
  <p class="portal-desc">
    Search, estimate, and compare commercial airline ticket pricing across primary domestic routes based on verified historical flight itineraries.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
avg_national_inr = 9087.0

with k1:
    st.markdown("""
    <div class="kpi-box">
      <div class="kpi-val">10,683</div>
      <div class="kpi-lbl">Verified Flight Records</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown("""
    <div class="kpi-box">
      <div class="kpi-val">10 Folds</div>
      <div class="kpi-lbl">Validation Protocol</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown("""
    <div class="kpi-box">
      <div class="kpi-val">5 Models</div>
      <div class="kpi-lbl">Benchmarked Models</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-val">{curr_sym}{(avg_national_inr * fx_rate):,.0f}</div>
      <div class="kpi-lbl">Average Route Fare</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["Flight Fare Calculator", "Route & Carrier Trends", "Model Performance & Benchmarks"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — FARE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # ── Presets ───────────────────────────────────────────────────────────────
    st.markdown("##### Frequent Route Templates")

    if 'f_airline' not in st.session_state: st.session_state.f_airline = 'IndiGo'
    if 'f_source' not in st.session_state: st.session_state.f_source = 'Banglore'
    if 'f_destination' not in st.session_state: st.session_state.f_destination = 'Delhi'
    if 'f_stops' not in st.session_state: st.session_state.f_stops = 'Non-stop (Direct)'
    if 'f_dur_h' not in st.session_state: st.session_state.f_dur_h = 2
    if 'f_dur_m' not in st.session_state: st.session_state.f_dur_m = 50
    if 'f_dep_h' not in st.session_state: st.session_state.f_dep_h = 10
    if 'f_dep_m' not in st.session_state: st.session_state.f_dep_m = 30
    if 'f_arr_h' not in st.session_state: st.session_state.f_arr_h = 13
    if 'f_arr_m' not in st.session_state: st.session_state.f_arr_m = 20

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("BLR ➔ DEL (IndiGo Direct)", use_container_width=True):
            st.session_state.f_airline = 'IndiGo'
            st.session_state.f_source = 'Banglore'
            st.session_state.f_destination = 'Delhi'
            st.session_state.f_stops = 'Non-stop (Direct)'
            st.session_state.f_dur_h = 2
            st.session_state.f_dur_m = 50
            st.session_state.f_dep_h = 10
            st.session_state.f_dep_m = 30
            st.session_state.f_arr_h = 13
            st.session_state.f_arr_m = 20
            st.rerun()

    with p2:
        if st.button("DEL ➔ COK (Air India 1-Stop)", use_container_width=True):
            st.session_state.f_airline = 'Air India'
            st.session_state.f_source = 'Delhi'
            st.session_state.f_destination = 'Cochin'
            st.session_state.f_stops = '1 Stop'
            st.session_state.f_dur_h = 5
            st.session_state.f_dur_m = 30
            st.session_state.f_dep_h = 8
            st.session_state.f_dep_m = 0
            st.session_state.f_arr_h = 13
            st.session_state.f_arr_m = 30
            st.rerun()

    with p3:
        if st.button("CCU ➔ BLR (Vistara 1-Stop)", use_container_width=True):
            st.session_state.f_airline = 'Vistara'
            st.session_state.f_source = 'Kolkata'
            st.session_state.f_destination = 'Banglore'
            st.session_state.f_stops = '1 Stop'
            st.session_state.f_dur_h = 7
            st.session_state.f_dur_m = 15
            st.session_state.f_dep_h = 14
            st.session_state.f_dep_m = 10
            st.session_state.f_arr_h = 21
            st.session_state.f_arr_m = 25
            st.rerun()

    with p4:
        if st.button("MAA ➔ CCU (SpiceJet Direct)", use_container_width=True):
            st.session_state.f_airline = 'SpiceJet'
            st.session_state.f_source = 'Chennai'
            st.session_state.f_destination = 'Kolkata'
            st.session_state.f_stops = 'Non-stop (Direct)'
            st.session_state.f_dur_h = 2
            st.session_state.f_dur_m = 20
            st.session_state.f_dep_h = 6
            st.session_state.f_dep_m = 45
            st.session_state.f_arr_h = 9
            st.session_state.f_arr_m = 5
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Flight Search Form ────────────────────────────────────────────────────
    with st.form("flight_estimator_form"):
        st.markdown("##### Flight Schedule & Route Details")

        f_r1_1, f_r1_2, f_r1_3 = st.columns(3)
        with f_r1_1:
            air_idx = AIRLINES.index(st.session_state.f_airline) if st.session_state.f_airline in AIRLINES else 0
            airline = st.selectbox("Operating Airline", AIRLINES, index=air_idx)
        with f_r1_2:
            src_idx = SOURCES.index(st.session_state.f_source) if st.session_state.f_source in SOURCES else 0
            source = st.selectbox("Departure Airport (Origin)", SOURCES, index=src_idx)
        with f_r1_3:
            dest_idx = DESTINATIONS.index(st.session_state.f_destination) if st.session_state.f_destination in DESTINATIONS else 0
            destination = st.selectbox("Arrival Airport (Destination)", DESTINATIONS, index=dest_idx)

        f_r2_1, f_r2_2, f_r2_3 = st.columns(3)
        with f_r2_1:
            stops_keys = list(STOPS_MAP.keys())
            st_idx = stops_keys.index(st.session_state.f_stops) if st.session_state.f_stops in stops_keys else 0
            stops_label = st.selectbox("Route Routing", stops_keys, index=st_idx)
            total_stops = STOPS_MAP[stops_label]
        with f_r2_2:
            journey_date = st.date_input("Travel Date", value=date(2024, 6, 15))
        with f_r2_3:
            dc1, dc2 = st.columns(2)
            with dc1:
                duration_hours = st.number_input("Duration Hours", 0, 40, int(st.session_state.f_dur_h))
            with dc2:
                duration_minutes = st.number_input("Minutes", 0, 59, int(st.session_state.f_dur_m), step=5)

        f_r3_1, f_r3_2 = st.columns(2)
        with f_r3_1:
            tc1, tc2 = st.columns(2)
            with tc1:
                dep_hour = st.number_input("Departure Hour (00-23)", 0, 23, int(st.session_state.f_dep_h))
            with tc2:
                dep_min = st.number_input("Departure Minute", 0, 59, int(st.session_state.f_dep_m), step=5)
        with f_r3_2:
            ac1, ac2 = st.columns(2)
            with ac1:
                arr_hour = st.number_input("Arrival Hour (00-23)", 0, 23, int(st.session_state.f_arr_h))
            with ac2:
                arr_min = st.number_input("Arrival Minute", 0, 59, int(st.session_state.f_arr_m), step=5)

        submitted = st.form_submit_button("Search Estimated Fare", use_container_width=True)

    # ── Calculation & Display ─────────────────────────────────────────────────
    if submitted or 'last_inr_fare' in st.session_state:
        if model_bundle is not None:
            le_dict = model_bundle.get('le_dict', {})
            features = model_bundle.get('features', [])
            model = model_bundle['model']

            def safe_encode(le, val):
                if le is not None and hasattr(le, 'classes_') and val in le.classes_:
                    return int(le.transform([val])[0])
                return 0

            air_enc = safe_encode(le_dict.get('Airline'), airline)
            src_enc = safe_encode(le_dict.get('Source'), source)
            dest_enc = safe_encode(le_dict.get('Destination'), destination)

            total_mins = (duration_hours * 60) + duration_minutes
            j_day = journey_date.day
            j_month = journey_date.month
            j_dayofweek = journey_date.weekday()

            row_dict = {
                'Airline': air_enc,
                'Source': src_enc,
                'Destination': dest_enc,
                'Total_Stops': total_stops,
                'Journey_Day': j_day,
                'Journey_Month': j_month,
                'Journey_DayOfWeek': j_dayofweek,
                'Dep_Hour': dep_hour,
                'Dep_Min': dep_min,
                'Arrival_Hour': arr_hour,
                'Arrival_Min': arr_min,
                'Duration_Mins': total_mins,
            }

            if features:
                input_df = pd.DataFrame([row_dict])[features]
                pred_raw = float(model.predict(input_df)[0])
            else:
                input_arr = np.array([list(row_dict.values())])
                pred_raw = float(model.predict(input_arr)[0])

            fare_inr = max(1500.0, float(pred_raw))
            st.session_state.last_inr_fare = fare_inr

            # Professional Fare Tiers in INR (Sandy & Black Palette - NO NEON)
            if fare_inr < 5000:
                tier = "Saver Economy"
                tier_bg, tier_fg = "#222019", "#cbb592"
                advisory = "Highly competitive rate. Ideal for budget travelers; check cabin baggage limits."
            elif fare_inr < 8500:
                tier = "Standard Economy"
                tier_bg, tier_fg = "#26231c", "#dfcfbe"
                advisory = "Fair market price for domestic routes. Regular fare with standard seating."
            elif fare_inr < 13000:
                tier = "Flexible Economy"
                tier_bg, tier_fg = "#2c261d", "#e8caa0"
                advisory = "Elevated fare due to high route demand or proximity to departure date."
            elif fare_inr < 20000:
                tier = "Peak Demand"
                tier_bg, tier_fg = "#2c201d", "#dfa497"
                advisory = "Peak pricing. Flexible dates or early morning flights may provide substantial savings."
            else:
                tier = "Business / Luxury Class"
                tier_bg, tier_fg = "#29212c", "#d0aede"
                advisory = "Premium booking tier with priority services, flexible rescheduling, and extra baggage."

            disp_amount = fare_inr * fx_rate
            diff_pct = ((fare_inr - avg_national_inr) / avg_national_inr) * 100
            diff_sign = "lower" if diff_pct < 0 else "higher"
            cost_hr = (fare_inr / (total_mins / 60)) if total_mins > 0 else 0

            st.markdown("<br>", unsafe_allow_html=True)

            col_res, col_scale = st.columns([1.1, 1])

            with col_res:
                st.markdown(f"""
                <div class="ticket-card">
                  <div class="ticket-route">
                    <div>
                      <div class="route-city">{source}</div>
                      <div class="route-airline">Origin Airport</div>
                    </div>
                    <div class="flight-icon-center">✈</div>
                    <div style="text-align:right;">
                      <div class="route-city">{destination}</div>
                      <div class="route-airline">Destination Airport</div>
                    </div>
                  </div>
                  <div style="text-align:center;">
                    <div class="ticket-price-sub">Estimated Total Base Fare ({currency.split('—')[0].strip()})</div>
                    <div class="ticket-price-val">{curr_sym}{disp_amount:,.0f}</div>
                    <div style="font-size:0.85rem; color:#8f8375; font-weight:600;">Carrier: {airline} · {total_stops} Stop{'s' if total_stops != 1 else ''}</div>
                    <div class="deal-badge" style="background:{tier_bg}; color:{tier_fg}; border:1px solid {tier_fg}44;">
                      {tier.upper()}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                r_p1, r_p2, r_p3 = st.columns(3)
                with r_p1:
                    st.markdown(f"""
                    <div class="pill-card">
                      <div class="pill-card-val">{curr_sym}{(cost_hr * fx_rate):,.0f}</div>
                      <div class="pill-card-lbl">Fare / Flight Hour</div>
                    </div>""", unsafe_allow_html=True)
                with r_p2:
                    st.markdown(f"""
                    <div class="pill-card">
                      <div class="pill-card-val">{abs(diff_pct):.1f}% {diff_sign}</div>
                      <div class="pill-card-lbl">vs Route Average</div>
                    </div>""", unsafe_allow_html=True)
                with r_p3:
                    st.markdown(f"""
                    <div class="pill-card">
                      <div class="pill-card-val">{duration_hours}h {duration_minutes}m</div>
                      <div class="pill-card-lbl">Flight Duration</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#191816; border:1px solid #332f28; border-left:4px solid #cbb592; padding:12px 14px; border-radius:6px; font-size:0.86rem; color:#dfcfbe;">
                  <strong style="color:#f5efe6;">Booking Advisory:</strong> {advisory}
                </div>""", unsafe_allow_html=True)

            with col_scale:
                # Matte Sandy & Black Gauge in INR (No Neon)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=fare_inr,
                    title={"text": "Market Fare Benchmark (INR ₹)", "font": {"color": "#8f8375", "size": 13}},
                    number={"prefix": "₹", "valueformat": ",.0f", "font": {"color": "#f5efe6", "size": 30}},
                    delta={"reference": avg_national_inr, "relative": False,
                           "decreasing": {"color": "#cbb592"}, "increasing": {"color": "#dfa497"}},
                    gauge={
                        "axis": {"range": [0, 35000], "tickcolor": "#4a4235", "tickfont": {"color": "#8f8375"}},
                        "bar": {"color": "#cbb592", "thickness": 0.28},
                        "bgcolor": "#161513",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 5000], "color": "#1f1d18"},
                            {"range": [5000, 8500], "color": "#28241e"},
                            {"range": [8500, 13000], "color": "#332d24"},
                            {"range": [13000, 20000], "color": "#3d3429"},
                            {"range": [20000, 35000], "color": "#4a3e2f"},
                        ],
                        "threshold": {
                            "line": {"color": "#f5efe6", "width": 2},
                            "thickness": 0.8,
                            "value": fare_inr,
                        },
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#191816",
                    plot_bgcolor="#191816",
                    font={"color": "#dfcfbe"},
                    height=280,
                    margin=dict(l=15, r=15, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#8f8375; margin-bottom:6px;'>Domestic Fare Brackets (INR):</div>", unsafe_allow_html=True)
                brackets = [
                    ("Saver Economy", "< ₹5,000", "#cbb592"),
                    ("Standard Economy", "₹5,000 – ₹8,500", "#dfcfbe"),
                    ("Flexible Economy", "₹8,500 – ₹13,000", "#e8caa0"),
                    ("Peak Demand", "₹13,000 – ₹20,000", "#dfa497"),
                    ("Business / Luxury", "> ₹20,000", "#d0aede"),
                ]
                for b_name, b_rng, b_col in brackets:
                    is_current = (b_name == tier)
                    bg_style = f"background:#26231e; border:1px solid {b_col};" if is_current else "background:#191816; border:1px solid #2d2a23;"
                    curr_mark = f"<span style='color:{b_col}; font-weight:700;'>Selected Route</span>" if is_current else ""
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:6px 12px; border-radius:6px; margin:4px 0; font-size:0.8rem; {bg_style}">
                      <div style="display:flex; align-items:center; gap:8px;">
                        <span style="width:8px; height:8px; border-radius:50%; background:{b_col};"></span>
                        <span style="color:#dfcfbe; font-weight:600;">{b_name}</span>
                        <span style="color:#8f8375;">({b_rng})</span>
                      </div>
                      <div>{curr_mark}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROUTE & MARKET DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is not None:
        st.markdown("##### Route Pricing & Carrier Market Distribution")

        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            sel_airlines = st.multiselect("Carrier Filter:", sorted(df['Airline'].dropna().unique()), default=[])
        with c_f2:
            sel_sources = st.multiselect("Origin City Filter:", sorted(df['Source'].dropna().unique()), default=[])
        with c_f3:
            sel_stops = st.multiselect("Stops Filter:", sorted(df['Total_Stops'].dropna().unique()), default=[])

        f_df = df.copy()
        if sel_airlines:
            f_df = f_df[f_df['Airline'].isin(sel_airlines)]
        if sel_sources:
            f_df = f_df[f_df['Source'].isin(sel_sources)]
        if sel_stops:
            f_df = f_df[f_df['Total_Stops'].isin(sel_stops)]

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Available Flights", f"{len(f_df):,}")
        m2.metric("Active Carriers", str(f_df['Airline'].nunique()))
        m3.metric("Average Fare (INR)", f"₹{f_df['Price'].mean():,.0f}")
        m4.metric("Fare Range (INR)", f"₹{f_df['Price'].min():,} – ₹{f_df['Price'].max():,}")

        st.markdown("<br>", unsafe_allow_html=True)
        display_cols = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration', 'Price']
        st.dataframe(f_df[display_cols].rename(columns={'Price': 'Price (₹)'}).head(12), use_container_width=True, height=270)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Market Visualizations (INR ₹)")

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            fig_hist = px.histogram(
                f_df, x='Price', nbins=40,
                title="Fare Distribution (₹)",
                labels={'Price': 'Ticket Price (₹)'},
                color_discrete_sequence=['#cbb592']
            )
            fig_hist.update_layout(paper_bgcolor='#191816', plot_bgcolor='#191816', font_color='#dfcfbe')
            fig_hist.update_xaxes(gridcolor='#2d2a23')
            fig_hist.update_yaxes(gridcolor='#2d2a23')
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_c2:
            carrier_avg = f_df.groupby('Airline')['Price'].mean().reset_index().sort_values('Price', ascending=False).head(8)
            fig_bar = px.bar(
                carrier_avg, x='Airline', y='Price',
                title="Top 8 Airlines by Average Ticket Fare (₹)",
                labels={'Price': 'Average Fare (₹)'},
                color_discrete_sequence=['#b59e7a']
            )
            fig_bar.update_layout(paper_bgcolor='#191816', plot_bgcolor='#191816', font_color='#dfcfbe')
            fig_bar.update_xaxes(gridcolor='#2d2a23')
            fig_bar.update_yaxes(gridcolor='#2d2a23')
            st.plotly_chart(fig_bar, use_container_width=True)

        chart_c3, chart_c4 = st.columns(2)
        with chart_c3:
            stops_avg = f_df.groupby('Total_Stops')['Price'].mean().reset_index()
            fig_stops = px.bar(
                stops_avg, x='Total_Stops', y='Price',
                title="Average Fare by Route Stops (₹)",
                labels={'Price': 'Average Fare (₹)', 'Total_Stops': 'Stops'},
                color_discrete_sequence=['#9e8763']
            )
            fig_stops.update_layout(paper_bgcolor='#191816', plot_bgcolor='#191816', font_color='#dfcfbe')
            fig_stops.update_xaxes(gridcolor='#2d2a23')
            fig_stops.update_yaxes(gridcolor='#2d2a23')
            st.plotly_chart(fig_stops, use_container_width=True)

        with chart_c4:
            src_dest = pd.crosstab(f_df['Source'], f_df['Destination'], values=f_df['Price'], aggfunc='mean').fillna(0)
            fig_heat = px.imshow(
                src_dest, text_auto=",.0f",
                title="Route Fare Matrix (₹)",
                color_continuous_scale=[[0, '#191816'], [0.5, '#5e503f'], [1, '#cbb592']]
            )
            fig_heat.update_layout(paper_bgcolor='#191816', plot_bgcolor='#191816', font_color='#dfcfbe')
            st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("##### 10-Fold Cross-Validation Performance Leaderboard")
    st.markdown("""
    <div style="background:#191816; border:1px solid #332f28; border-radius:8px; padding:12px 16px; margin-bottom:1.2rem; font-size:0.86rem; color:#a89d8f;">
      Evaluation across 10,682 verified flight instances using 10-fold cross-validation. All error metrics reported in Indian Rupee (INR ₹).
    </div>
    """, unsafe_allow_html=True)

    pod1, pod2, pod3 = st.columns(3)
    with pod1:
        st.markdown("""
        <div class="kpi-box" style="border-top:3px solid #cbb592;">
          <div style="display:inline-block; background:#26231e; color:#cbb592; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px; margin-bottom:4px; border:1px solid #4a4235;">Rank #1 · Selected Model</div>
          <div style="font-weight:700; color:#f5efe6; font-size:1.05rem; margin-top:2px;">Random Forest Regressor</div>
          <div style="color:#cbb592; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: ₹2,051</div>
          <div style="font-size:0.75rem; color:#8f8375;">R²: 0.8011 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod2:
        st.markdown("""
        <div class="kpi-box" style="border-top:3px solid #524a3e;">
          <div style="display:inline-block; background:#201e1b; color:#a89d8f; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px; margin-bottom:4px; border:1px solid #38332b;">Rank #2</div>
          <div style="font-weight:700; color:#f5efe6; font-size:1.05rem; margin-top:2px;">Extra Trees Regressor</div>
          <div style="color:#dfcfbe; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: ₹2,251</div>
          <div style="font-size:0.75rem; color:#8f8375;">R²: 0.7601 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod3:
        st.markdown("""
        <div class="kpi-box" style="border-top:3px solid #524a3e;">
          <div style="display:inline-block; background:#201e1b; color:#a89d8f; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px; margin-bottom:4px; border:1px solid #38332b;">Rank #3</div>
          <div style="font-weight:700; color:#f5efe6; font-size:1.05rem; margin-top:2px;">SVR (RBF Kernel)</div>
          <div style="color:#dfcfbe; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: ₹3,155</div>
          <div style="font-size:0.75rem; color:#8f8375;">R²: 0.5314 · Scaled C=100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Table
    lb_df = pd.DataFrame({
        "Rank": ["1st", "2nd", "3rd", "4th", "5th"],
        "Algorithm": ["Random Forest Regressor", "Extra Trees Regressor", "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "Category": ["Bagged Decision Trees", "Extremely Randomized Trees", "Support Vector Machine", "L2 Regularized Linear", "L1 Regularized Linear"],
        "10-Fold CV RMSE (INR)": ["₹2,050.99", "₹2,250.72", "₹3,154.68", "₹3,445.00", "₹3,445.00"],
        "R² Score": ["0.8011", "0.7601", "0.5314", "0.4408", "0.4408"],
        "Configuration": [
            "n_estimators=100, max_features='sqrt'",
            "n_estimators=100, max_features='sqrt'",
            "C=100, gamma='scale', RBF",
            "alpha=1.0, StandardScaler",
            "alpha=0.1, max_iter=10000",
        ]
    })
    st.dataframe(lb_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 10-Fold CV RMSE Comparison (Lower is Better)")

    sim_df = pd.DataFrame({
        "Model": ["Random Forest Regressor", "Extra Trees Regressor", "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "RMSE_INR": [2050.99, 2250.72, 3154.68, 3445.00, 3445.00],
    })

    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        y=sim_df["Model"],
        x=sim_df["RMSE_INR"],
        orientation='h',
        marker_color=['#cbb592', '#b89f78', '#9e8562', '#78664f', '#574938'],
        text=[f"₹{v:,.0f}" for v in sim_df["RMSE_INR"]],
        textposition='outside',
    ))
    fig_rmse.update_layout(
        paper_bgcolor='#191816',
        plot_bgcolor='#191816',
        font_color='#dfcfbe',
        xaxis_title="Cross-Validation RMSE in Indian Rupee (INR ₹)",
        yaxis=dict(autorange='reversed'),
        height=300,
        margin=dict(l=10, r=80, t=10, b=40),
    )
    fig_rmse.update_xaxes(gridcolor='#2d2a23')
    st.plotly_chart(fig_rmse, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7a7063; font-size:0.78rem; padding: 0.8rem 0 1.2rem;">
  <strong>SkyFare Domestic Flight Price Index</strong><br>
  Calibrated on 10,683 Domestic Flights · Currency: Indian Rupee (INR ₹)
</div>
""", unsafe_allow_html=True)
