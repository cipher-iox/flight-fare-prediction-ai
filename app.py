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

# ── Exchange Rates (Base dataset in INR) ───────────────────────────────────────
INR_TO_PKR = 3.32  # 1 INR = ~3.32 PKR

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyFare | Commercial Flight Price Estimator",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional Aviation Styling (Refined Navy & Slate Palette) ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; 
}

/* Background - Clean Dark Aviation Slate */
.stApp {
    background-color: #0c121e;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }

/* Clean Aviation Banner */
.portal-header {
    background: #131d2e;
    border: 1px solid #1e2c42;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.portal-badge {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #2b4c7e;
    padding: 3px 12px;
    border-radius: 6px;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.portal-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.portal-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    max-width: 720px;
}

/* KPI Summary Cards */
.kpi-box {
    background: #131d2e;
    border: 1px solid #1e2c42;
    border-radius: 10px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    transition: border-color 0.2s ease;
}
.kpi-box:hover {
    border-color: #2563eb;
}
.kpi-val {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.5px;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
    margin-top: 4px;
}

/* Flight Ticket Card Aesthetic */
.ticket-card {
    background: #131d2e;
    border: 1px solid #1e2c42;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}
.ticket-route {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px dashed #24354f;
    padding-bottom: 1.2rem;
    margin-bottom: 1.2rem;
}
.route-city {
    font-size: 1.3rem;
    font-weight: 800;
    color: #f8fafc;
}
.route-airline {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
}
.flight-icon-center {
    font-size: 1.4rem;
    color: #3b82f6;
}
.ticket-price-val {
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1;
    margin: 0.4rem 0 0.2rem;
    letter-spacing: -1px;
}
.ticket-price-sub {
    font-size: 0.82rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* Pill Badges */
.deal-badge {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    margin-top: 1rem;
}

.pill-card {
    background: #0d1522;
    border: 1px solid #1e2c42;
    border-radius: 8px;
    padding: 0.75rem 0.9rem;
    text-align: center;
}
.pill-card-val {
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
}
.pill-card-lbl {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    font-weight: 600;
    margin-top: 2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a0f1a;
    border-right: 1px solid #1a2536;
}

/* Corporate Buttons */
div.stButton > button:first-child {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    transition: background-color 0.2s ease;
}
div.stButton > button:first-child:hover {
    background-color: #1d4ed8;
    color: #ffffff;
}

/* Clean Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background-color: #111a28;
    padding: 5px;
    border-radius: 8px;
    border: 1px solid #1e2c42;
}
.stTabs [data-baseweb="tab"] {
    height: 40px;
    border-radius: 6px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 0 16px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #ffffff !important;
    border: 1px solid #334155 !important;
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
        df_raw = pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        df_raw = pd.read_excel(xlsx_path)
    else:
        return None
    # Add PKR converted price column
    df_raw['Price_PKR'] = (df_raw['Price'] * INR_TO_PKR).round()
    return df_raw

df = load_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1.2rem;">
      <div style="font-size:1.6rem; color:#3b82f6;">✈️</div>
      <div>
        <div style="font-weight:800; font-size:1.15rem; color:#f8fafc; font-family:'Space Grotesk';">SkyFare Portal</div>
        <div style="font-size:0.75rem; color:#64748b;">Flight Price Analytics</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model_bundle:
        st.markdown("""
        <div style="background:#0e2017; border:1px solid #1b4d2e; border-radius:6px; padding:6px 10px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e;"></span>
          <span style="font-size:0.8rem; color:#4ade80; font-weight:600;">Pricing Engine Online</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#221010; border:1px solid #4a1d1d; border-radius:6px; padding:6px 10px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444;"></span>
          <span style="font-size:0.8rem; color:#f87171; font-weight:600;">Model Offline</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Currency Settings")
    currency = st.selectbox("Display Currency", [
        "PKR (Rs.) — Pakistani Rupee",
        "USD ($) — US Dollar",
        "EUR (€) — Euro",
        "GBP (£) — British Pound",
        "AED (AED) — UAE Dirham",
        "SAR (SAR) — Saudi Riyal"
    ], index=0)

    # Conversion rates relative to PKR
    curr_rates = {
        "PKR (Rs.) — Pakistani Rupee": (1.0, "Rs. "),
        "USD ($) — US Dollar": (1 / 278.5, "$"),
        "EUR (€) — Euro": (1 / 302.0, "€"),
        "GBP (£) — British Pound": (1 / 355.0, "£"),
        "AED (AED) — UAE Dirham": (1 / 75.8, "AED "),
        "SAR (SAR) — Saudi Riyal": (1 / 74.2, "SAR ")
    }
    fx_rate, curr_sym = curr_rates[currency]

    st.markdown("---")
    st.markdown("#### System Overview")
    st.markdown("""
    - **Course:** AIC354 ML Fundamentals
    - **Methodology:** 10-Fold Cross-Validation
    - **Evaluated Dataset:** 10,683 Commercial Flights
    - **Standard Currency:** Pakistani Rupee (PKR)
    """)

    if model_bundle:
        st.markdown("---")
        st.markdown("#### Active Estimator")
        st.markdown(f"**Model:** `{model_bundle.get('model_name', 'Random Forest Regressor')}`")
        st.markdown(f"**Input Attributes:** `{len(model_bundle.get('features', []))}`")
        if 'rmse' in model_bundle:
            rmse_pkr = model_bundle['rmse'] * INR_TO_PKR
            st.markdown(f"**CV Error (RMSE):** `Rs. {rmse_pkr:,.0f}`")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER / BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="portal-header">
  <div class="portal-badge">Commercial Aviation Analytics · Lab 3</div>
  <h1 class="portal-title">Flight Fare Estimation & Price Index</h1>
  <p class="portal-desc">
    Forecast commercial airline ticket pricing across primary domestic routes using 10-Fold cross-validated ensemble learning models calibrated on 10,683 verified travel itineraries.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
avg_national_pkr = 9087 * INR_TO_PKR

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
      <div class="kpi-lbl">Evaluated Algorithms</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-val">{curr_sym}{(avg_national_pkr * fx_rate):,.0f}</div>
      <div class="kpi-lbl">Average Ticket Fare</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["✈️ Fare Estimator", "📊 Route & Market Data", "📈 Model Benchmarks"])

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

        submitted = st.form_submit_button("Calculate Estimated Fare", use_container_width=True)

    # ── Calculation & Display ─────────────────────────────────────────────────
    if submitted or 'last_pkr_fare' in st.session_state:
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

            # Convert to PKR
            fare_pkr = max(5000.0, float(pred_raw * INR_TO_PKR))
            st.session_state.last_pkr_fare = fare_pkr

            # Professional Fare Tiers in PKR
            if fare_pkr < 16500:
                tier = "Saver Economy"
                tier_bg, tier_fg = "#0f2e1e", "#34d399"
                advisory = "Highly competitive rate. Ideal for budget travelers; check cabin baggage limits."
            elif fare_pkr < 28000:
                tier = "Standard Economy"
                tier_bg, tier_fg = "#14253d", "#60a5fa"
                advisory = "Fair market price for domestic routes. Regular fare with standard seating."
            elif fare_pkr < 45000:
                tier = "Flexible Economy"
                tier_bg, tier_fg = "#2e210f", "#fbbf24"
                advisory = "Elevated fare due to high route demand or proximity to departure date."
            elif fare_pkr < 70000:
                tier = "Premium / High Demand"
                tier_bg, tier_fg = "#331616", "#f87171"
                advisory = "Peak pricing. Flexible dates or early morning flights may provide substantial savings."
            else:
                tier = "Business / Luxury Class"
                tier_bg, tier_fg = "#261738", "#c084fc"
                advisory = "Premium booking tier with priority services, flexible rescheduling, and extra baggage."

            disp_amount = fare_pkr * fx_rate
            diff_pct = ((fare_pkr - avg_national_pkr) / avg_national_pkr) * 100
            diff_sign = "lower" if diff_pct < 0 else "higher"
            cost_hr = (fare_pkr / (total_mins / 60)) if total_mins > 0 else 0

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
                    <div class="ticket-price-sub">Estimated Total Ticket Fare ({currency.split('—')[0].strip()})</div>
                    <div class="ticket-price-val">{curr_sym}{disp_amount:,.0f}</div>
                    <div style="font-size:0.85rem; color:#94a3b8; font-weight:600;">Carrier: {airline} · {total_stops} Stop{'s' if total_stops != 1 else ''}</div>
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
                      <div class="pill-card-lbl">vs National Route Avg</div>
                    </div>""", unsafe_allow_html=True)
                with r_p3:
                    st.markdown(f"""
                    <div class="pill-card">
                      <div class="pill-card-val">{duration_hours}h {duration_minutes}m</div>
                      <div class="pill-card-lbl">Flight Time</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#131d2e; border-left:3px solid #2563eb; padding:12px 14px; border-radius:6px; font-size:0.86rem; color:#cbd5e1;">
                  <strong>Booking Advisory:</strong> {advisory}
                </div>""", unsafe_allow_html=True)

            with col_scale:
                # Clean Price Gauge in PKR
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=fare_pkr,
                    title={"text": "Market Fare Benchmark (PKR)", "font": {"color": "#94a3b8", "size": 13}},
                    number={"prefix": "Rs. ", "valueformat": ",.0f", "font": {"color": "#f8fafc", "size": 30}},
                    delta={"reference": avg_national_pkr, "relative": False,
                           "decreasing": {"color": "#22c55e"}, "increasing": {"color": "#ef4444"}},
                    gauge={
                        "axis": {"range": [0, 110000], "tickcolor": "#334155", "tickfont": {"color": "#64748b"}},
                        "bar": {"color": "#2563eb", "thickness": 0.28},
                        "bgcolor": "#131d2e",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 16500], "color": "#0f2e1e"},
                            {"range": [16500, 28000], "color": "#14253d"},
                            {"range": [28000, 45000], "color": "#2e210f"},
                            {"range": [45000, 70000], "color": "#331616"},
                            {"range": [70000, 110000], "color": "#261738"},
                        ],
                        "threshold": {
                            "line": {"color": "#3b82f6", "width": 3},
                            "thickness": 0.8,
                            "value": fare_pkr,
                        },
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc"},
                    height=280,
                    margin=dict(l=15, r=15, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("<div style='font-size:0.8rem; font-weight:700; color:#94a3b8; margin-bottom:6px;'>Domestic Fare Brackets (PKR):</div>", unsafe_allow_html=True)
                brackets = [
                    ("Saver Economy", "< Rs. 16,500", "#34d399"),
                    ("Standard Economy", "Rs. 16,500 – 28,000", "#60a5fa"),
                    ("Flexible Economy", "Rs. 28,000 – 45,000", "#fbbf24"),
                    ("Premium Demand", "Rs. 45,000 – 70,000", "#f87171"),
                    ("Business / Luxury", "> Rs. 70,000", "#c084fc"),
                ]
                for b_name, b_rng, b_col in brackets:
                    is_current = (b_name == tier)
                    bg_style = f"background:#1e293b; border:1px solid {b_col};" if is_current else "background:#0f172a;"
                    curr_mark = f"<span style='color:{b_col}; font-weight:700;'>Selected Route</span>" if is_current else ""
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center; padding:5px 10px; border-radius:6px; margin:3px 0; font-size:0.8rem; {bg_style}">
                      <div style="display:flex; align-items:center; gap:8px;">
                        <span style="width:8px; height:8px; border-radius:50%; background:{b_col};"></span>
                        <span style="color:#e2e8f0; font-weight:600;">{b_name}</span>
                        <span style="color:#64748b;">({b_rng})</span>
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
        m3.metric("Average Fare (PKR)", f"Rs. {f_df['Price_PKR'].mean():,.0f}")
        m4.metric("Fare Range (PKR)", f"Rs. {f_df['Price_PKR'].min():,.0f} – {f_df['Price_PKR'].max():,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        display_cols = ['Airline', 'Source', 'Destination', 'Total_Stops', 'Duration', 'Price_PKR']
        st.dataframe(f_df[display_cols].rename(columns={'Price_PKR': 'Price (PKR)'}).head(12), use_container_width=True, height=270)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Market Visualizations (All values in PKR)")

        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            fig_hist = px.histogram(
                f_df, x='Price_PKR', nbins=40,
                title="Fare Distribution (PKR)",
                labels={'Price_PKR': 'Ticket Price (PKR)'},
                color_discrete_sequence=['#2563eb']
            )
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_hist, use_container_width=True)

        with chart_c2:
            carrier_avg = f_df.groupby('Airline')['Price_PKR'].mean().reset_index().sort_values('Price_PKR', ascending=False).head(8)
            fig_bar = px.bar(
                carrier_avg, x='Airline', y='Price_PKR',
                title="Top 8 Airlines by Average Ticket Fare (PKR)",
                labels={'Price_PKR': 'Average Fare (PKR)'},
                color_discrete_sequence=['#3b82f6']
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_bar, use_container_width=True)

        chart_c3, chart_c4 = st.columns(2)
        with chart_c3:
            stops_avg = f_df.groupby('Total_Stops')['Price_PKR'].mean().reset_index()
            fig_stops = px.bar(
                stops_avg, x='Total_Stops', y='Price_PKR',
                title="Average Fare by Route Stops (PKR)",
                labels={'Price_PKR': 'Average Fare (PKR)', 'Total_Stops': 'Stops'},
                color_discrete_sequence=['#0ea5e9']
            )
            fig_stops.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_stops, use_container_width=True)

        with chart_c4:
            src_dest = pd.crosstab(f_df['Source'], f_df['Destination'], values=f_df['Price_PKR'], aggfunc='mean').fillna(0)
            fig_heat = px.imshow(
                src_dest, text_auto=",.0f",
                title="Route Fare Matrix (PKR)",
                color_continuous_scale='Blues'
            )
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("##### 10-Fold Cross-Validation Performance Leaderboard")
    st.markdown("""
    <div style="background:#131d2e; border:1px solid #1e2c42; border-radius:8px; padding:12px 16px; margin-bottom:1.2rem; font-size:0.86rem; color:#94a3b8;">
      Evaluation across 10,682 verified flight instances using 10-fold cross-validation. All error metrics converted to Pakistani Rupee (PKR).
    </div>
    """, unsafe_allow_html=True)

    pod1, pod2, pod3 = st.columns(3)
    with pod1:
        st.markdown("""
        <div class="kpi-box" style="border-color:#2563eb; background:#0d1829;">
          <div style="font-size:1.6rem;">🥇</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.05rem; margin-top:2px;">Random Forest Regressor</div>
          <div style="color:#60a5fa; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: Rs. 6,809</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.8011 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod2:
        st.markdown("""
        <div class="kpi-box" style="border-color:#334155;">
          <div style="font-size:1.6rem;">🥈</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.05rem; margin-top:2px;">Extra Trees Regressor</div>
          <div style="color:#94a3b8; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: Rs. 7,472</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.7601 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod3:
        st.markdown("""
        <div class="kpi-box" style="border-color:#334155;">
          <div style="font-size:1.6rem;">🥉</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.05rem; margin-top:2px;">SVR (RBF Kernel)</div>
          <div style="color:#94a3b8; font-weight:800; font-size:1.5rem; margin:4px 0;">RMSE: Rs. 10,474</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.5314 · Scaled C=100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Table
    lb_df = pd.DataFrame({
        "Rank": ["1st", "2nd", "3rd", "4th", "5th"],
        "Algorithm": ["Random Forest Regressor", "Extra Trees Regressor", "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "Category": ["Bagged Decision Trees", "Extremely Randomized Trees", "Support Vector Machine", "L2 Regularized Linear", "L1 Regularized Linear"],
        "10-Fold CV RMSE (PKR)": ["Rs. 6,809.28", "Rs. 7,472.39", "Rs. 10,473.54", "Rs. 11,437.40", "Rs. 11,437.40"],
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
        "RMSE_PKR": [6809.28, 7472.39, 10473.54, 11437.40, 11437.40],
    })

    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        y=sim_df["Model"],
        x=sim_df["RMSE_PKR"],
        orientation='h',
        marker_color=['#2563eb', '#3b82f6', '#64748b', '#475569', '#334155'],
        text=[f"Rs. {v:,.0f}" for v in sim_df["RMSE_PKR"]],
        textposition='outside',
    ))
    fig_rmse.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis_title="Cross-Validation RMSE in Pakistani Rupee (PKR)",
        yaxis=dict(autorange='reversed'),
        height=300,
        margin=dict(l=10, r=80, t=10, b=40),
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.78rem; padding: 0.8rem 0 1.2rem;">
  <strong>SkyFare Commercial Airline Price Portal · AIC354 ML Fundamentals Lab 3</strong><br>
  Calibrated on 10,683 Domestic Flights · Currency: Pakistani Rupee (PKR)
</div>
""", unsafe_allow_html=True)
