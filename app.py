"""
Flight Fare Prediction — Streamlit Web Application
AIC354 Machine Learning Fundamentals | Assignment 3
Deploy: https://share.streamlit.io → connect this GitHub repo → set app.py as entry point
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
    page_title="Flight Fare Predictor | AI Airfare Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS & Modern Design System ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { 
    font-family: 'Plus Jakarta Sans', sans-serif; 
}

/* Background mesh */
.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.10) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.10) 0px, transparent 50%),
        radial-gradient(at 50% 100%, rgba(16, 185, 129, 0.06) 0px, transparent 50%);
}

#MainMenu, footer, header { visibility: hidden; }

/* Custom Hero Banner */
.hero-container {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2.2rem 2rem 1.8rem;
    text-align: center;
    backdrop-filter: blur(16px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: 0; left: 20%; right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #38bdf8, #818cf8, transparent);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
    line-height: 1.2;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.98rem;
    margin-top: 0.6rem;
    max-width: 680px;
    margin-left: auto;
    margin-right: auto;
}

/* Metric Cards */
.metric-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.2rem 1rem;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(56, 189, 248, 0.15);
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    opacity: 0.7;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f8fafc, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 5px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

/* Glass Card */
.glass-panel {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 18px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin-bottom: 1.2rem;
}

/* Result Box */
.result-hero {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(11, 15, 25, 0.95));
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
}
.result-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1;
    margin: 0.5rem 0 0.2rem;
}
.result-unit {
    font-size: 0.95rem;
    color: #94a3b8;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}
.result-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 9999px;
    font-size: 0.88rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 1rem;
}

/* Feature Badge */
.stat-pill {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.stat-pill-val {
    font-size: 1.25rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'Space Grotesk', sans-serif;
}
.stat-pill-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Form Submit Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.8rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
    box-shadow: 0 8px 20px -5px rgba(2, 132, 199, 0.5);
    transition: all 0.25s ease;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 12px 25px -4px rgba(2, 132, 199, 0.75);
    transform: translateY(-2px);
    color: white;
}

/* Custom Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(15, 23, 42, 0.6);
    padding: 6px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.06);
}
.stTabs [data-baseweb="tab"] {
    height: 44px;
    border-radius: 10px;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0 18px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(99, 102, 241, 0.15)) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    color: #38bdf8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Reference Options ─────────────────────────────────────────────────────────
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
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:1rem;">
      <div style="font-size:2rem;">✈️</div>
      <div>
        <div style="font-weight:800; font-size:1.15rem; color:#f8fafc; font-family:'Space Grotesk';">AeroPrice AI</div>
        <div style="font-size:0.75rem; color:#64748b;">Flight Fare Intelligence</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if model_bundle:
        st.markdown("""
        <div style="background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.3);
                    border-radius:10px; padding:8px 12px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#10b981; box-shadow:0 0 8px #10b981;"></span>
          <span style="font-size:0.82rem; color:#10b981; font-weight:600;">Model Loaded & Active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239, 68, 68, 0.1); border:1px solid rgba(239, 68, 68, 0.3);
                    border-radius:10px; padding:8px 12px; margin-bottom:1.2rem; display:flex; align-items:center; gap:8px;">
          <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#ef4444;"></span>
          <span style="font-size:0.82rem; color:#ef4444; font-weight:600;">Model Offline</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Assignment Context")
    st.markdown("""
    - **Course:** AIC354 ML Fundamentals
    - **Assignment:** Lab 3 (Model Comparison)
    - **Validation:** 10-Fold Cross-Validation
    - **Dataset:** 10,683 Flight Records (India)
    - **Deployment:** Streamlit Cloud
    """)

    st.markdown("---")
    st.markdown("### 🏆 Active Model Details")
    if model_bundle:
        st.markdown(f"**Algorithm:** `{model_bundle.get('model_name', 'Extra Trees Regressor')}`")
        st.markdown(f"**Features In:** `{len(model_bundle.get('features', []))}`")
        st.markdown(f"**Target:** `{model_bundle.get('target', 'Price')} (INR ₹)`")
        if 'rmse' in model_bundle:
            st.markdown(f"**Validation RMSE:** `₹{model_bundle['rmse']:,.2f}`")
    else:
        st.caption("No model bundle loaded.")

    st.markdown("---")
    st.markdown("### 💱 Currency Converter Toggle")
    currency = st.selectbox("Display Currency", ["INR (₹)", "USD ($)", "EUR (€)", "GBP (£)"])
    rates = {"INR (₹)": 1.0, "USD ($)": 1 / 83.5, "EUR (€)": 1 / 90.2, "GBP (£)": 1 / 105.4}
    curr_sym = {"INR (₹)": "₹", "USD ($)": "$", "EUR (€)": "€", "GBP (£)": "£"}[currency]
    rate = rates[currency]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER / HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
  <div class="hero-badge">⚡ AIC354 Machine Learning Fundamentals · Assignment 3</div>
  <h1 class="hero-title">Flight Fare Prediction AI System</h1>
  <p class="hero-sub">
    Estimate commercial airfare dynamically powered by 10-Fold Cross-Validated Machine Learning algorithms trained on 10,683 domestic flight records.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Hero KPI Cards ────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpis = [
    (k1, "10,683", "Verified Flight Records", "📊"),
    (k2, "10-Fold", "Cross-Validation Protocol", "🔁"),
    (k3, "5", "Evaluated ML Algorithms", "🤖"),
    (k4, "₹9,087", "Average Base Airfare", "🎯"),
]
for col, val, label, icon in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:1.4rem; margin-bottom:4px;">{icon}</div>
          <div class="metric-val">{val}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Predict Flight Fare", "📊 Flight Data Explorer", "🤖 ML Model Comparison"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if model_bundle is None:
        st.warning("""
        ⚠️ **Trained model bundle not found (`models/best_model.pkl`).**  
        Run the notebook or training script to generate and export `best_model.pkl`.
        """)
    else:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;
                    background:rgba(56, 189, 248, 0.06); border:1px solid rgba(56, 189, 248, 0.2);
                    padding:0.75rem 1.2rem; border-radius:12px;">
          <div style="font-size:0.9rem; color:#f0f6ff;">
            Engine Status: <strong style="color:#38bdf8;">Production Active</strong> &nbsp;·&nbsp;
            Estimator: <strong style="color:#818cf8;">{model_bundle.get('model_name', 'Extra Trees Regressor')}</strong>
          </div>
          <div style="font-size:0.8rem; color:#64748b; font-family:'JetBrains Mono';">10-Fold CV Optimized</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Presets ───────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>⚡ Quick Route Presets (Click to Auto-fill)</div>", unsafe_allow_html=True)

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
        if st.button("⚡ Metro Express (BLR ➔ DEL)", use_container_width=True):
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
        if st.button("👔 Business Corridor (DEL ➔ BOM)", use_container_width=True):
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
        if st.button("🌴 South Leisure (CCU ➔ COK)", use_container_width=True):
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
        if st.button("💸 Budget Coastal (MAA ➔ CCU)", use_container_width=True):
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
    st.markdown("<div class='section-header'>⚙️ Flight Route & Scheduling Parameters</div>", unsafe_allow_html=True)

    with st.form("flight_prediction_form"):
        r1_c1, r1_c2, r1_c3 = st.columns(3)
        with r1_c1:
            air_idx = AIRLINES.index(st.session_state.f_airline) if st.session_state.f_airline in AIRLINES else 0
            airline = st.selectbox("✈️ Airline Carrier", AIRLINES, index=air_idx)
        with r1_c2:
            src_idx = SOURCES.index(st.session_state.f_source) if st.session_state.f_source in SOURCES else 0
            source = st.selectbox("🛫 Source City", SOURCES, index=src_idx)
        with r1_c3:
            dest_idx = DESTINATIONS.index(st.session_state.f_destination) if st.session_state.f_destination in DESTINATIONS else 0
            destination = st.selectbox("🛬 Destination City", DESTINATIONS, index=dest_idx)

        r2_c1, r2_c2, r2_c3 = st.columns(3)
        with r2_c1:
            stops_keys = list(STOPS_MAP.keys())
            st_idx = stops_keys.index(st.session_state.f_stops) if st.session_state.f_stops in stops_keys else 0
            stops_label = st.selectbox("🛑 Flight Stops", stops_keys, index=st_idx)
            total_stops = STOPS_MAP[stops_label]
        with r2_c2:
            journey_date = st.date_input("📅 Date of Departure", value=date(2024, 5, 15))
        with r2_c3:
            dur_cols = st.columns(2)
            with dur_cols[0]:
                duration_hours = st.number_input("Hours", min_value=0, max_value=40, value=int(st.session_state.f_dur_h), step=1)
            with dur_cols[1]:
                duration_minutes = st.number_input("Minutes", min_value=0, max_value=59, value=int(st.session_state.f_dur_m), step=5)

        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            dep_cols = st.columns(2)
            with dep_cols[0]:
                dep_hour = st.number_input("Departure Hour (0-23)", 0, 23, int(st.session_state.f_dep_h))
            with dep_cols[1]:
                dep_min = st.number_input("Departure Minute (0-59)", 0, 59, int(st.session_state.f_dep_m), step=5)
        with r3_c2:
            arr_cols = st.columns(2)
            with arr_cols[0]:
                arr_hour = st.number_input("Arrival Hour (0-23)", 0, 23, int(st.session_state.f_arr_h))
            with arr_cols[1]:
                arr_min = st.number_input("Arrival Minute (0-59)", 0, 59, int(st.session_state.f_arr_m), step=5)

        submitted = st.form_submit_button("🔮 Calculate Estimated Flight Fare", use_container_width=True)

    # ── Prediction Execution ───────────────────────────────────────────────────
    if submitted or 'last_flight_pred' in st.session_state:
        if submitted and model_bundle is None:
            st.error("❌ Model artifact not loaded.")
        elif model_bundle is not None:
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

            total_duration_mins = (duration_hours * 60) + duration_minutes
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
                'Duration_Mins': total_duration_mins,
            }

            if features:
                input_df = pd.DataFrame([row_dict])[features]
                pred_raw = float(model.predict(input_df)[0])
            else:
                input_arr = np.array([list(row_dict.values())])
                pred_raw = float(model.predict(input_arr)[0])

            predicted_inr = max(1500.0, float(pred_raw))
            st.session_state.last_flight_pred = predicted_inr

            # Fare Tier Classification
            if predicted_inr < 5000:
                tier, tier_col, tier_icon, tier_label = "Super Saver / Budget", "#10b981", "🌿", "⭐⭐⭐⭐⭐ Outstanding Value"
                tip = "Excellent low-cost deal! Ideal for light travelers and quick domestic transfers."
            elif predicted_inr < 8500:
                tier, tier_col, tier_icon, tier_label = "Economy Standard", "#0284c7", "✈️", "⭐⭐⭐⭐ Standard Fare"
                tip = "Fair market price for domestic routes. Check baggage allowances and seat selection terms."
            elif predicted_inr < 13000:
                tier, tier_col, tier_icon, tier_label = "Flexible / Peak Economy", "#f59e0b", "⚠️", "⭐⭐⭐ Elevated Fare"
                tip = "Higher than median fare. Likely due to close booking date, peak departure slot, or layovers."
            elif predicted_inr < 20000:
                tier, tier_col, tier_icon, tier_label = "Premium / High Season", "#ef4444", "🔴", "⭐⭐ High Demand"
                tip = "Substantial price point. Consider shifting departure by 1–2 days or choosing early morning departures."
            else:
                tier, tier_col, tier_icon, tier_label = "Luxury / Business Class", "#8b5cf6", "💎", "⭐ Premium Cabin"
                tip = "Top-tier premium booking. Includes full flexibility, priority boarding, and premium seating."

            disp_fare = predicted_inr * rate
            cost_per_hr = (predicted_inr / (total_duration_mins / 60)) if total_duration_mins > 0 else 0
            national_avg = 9087.0
            diff_pct = ((predicted_inr - national_avg) / national_avg) * 100

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>📊 Airfare Valuation & Booking Assessment</div>", unsafe_allow_html=True)

            res_col, gauge_col = st.columns([1.1, 1])

            with res_col:
                st.markdown(f"""
                <div class="result-hero" style="border-color:{tier_col}; box-shadow: 0 0 40px {tier_col}22;">
                  <div style="font-size:2.8rem; margin-bottom:0.2rem;">{tier_icon}</div>
                  <div class="result-val" style="color:{tier_col};">{curr_sym}{disp_fare:,.0f}</div>
                  <div class="result-unit">{airline} · {source} ➔ {destination}</div>
                  <div class="result-badge" style="background:{tier_col}22; color:{tier_col}; border:1px solid {tier_col}55;">
                    {tier.upper()} · {tier_label}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                r1, r2, r3 = st.columns(3)
                with r1:
                    st.markdown(f"""
                    <div class="stat-pill">
                      <div class="stat-pill-val">{curr_sym}{(cost_per_hr * rate):,.0f}/hr</div>
                      <div class="stat-pill-label">Cost / Flight Hour</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class="stat-pill">
                      <div class="stat-pill-val">{total_stops} stop{'s' if total_stops != 1 else ''}</div>
                      <div class="stat-pill-label">Route Stops</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    diff_str = f"{abs(diff_pct):.1f}% {'lower' if diff_pct < 0 else 'higher'}"
                    diff_col = "#10b981" if diff_pct < 0 else "#ef4444"
                    st.markdown(f"""
                    <div class="stat-pill">
                      <div class="stat-pill-val" style="color:{diff_col};">{diff_str}</div>
                      <div class="stat-pill-label">vs National Avg</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03); border-left:4px solid {tier_col};
                            padding:0.9rem 1.1rem; border-radius:10px; font-size:0.88rem; color:#cbd5e1; line-height:1.5;">
                  <strong>💡 Traveler Recommendation:</strong> {tip}
                </div>""", unsafe_allow_html=True)

            with gauge_col:
                # Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=predicted_inr,
                    title={"text": "Airfare Price Scale (INR ₹)", "font": {"color": "#94a3b8", "size": 14, "family": "Plus Jakarta Sans"}},
                    number={"prefix": "₹", "valueformat": ",.0f", "font": {"color": tier_col, "size": 32, "family": "Space Grotesk"}},
                    delta={"reference": 9087, "relative": False,
                           "decreasing": {"color": "#10b981"},
                           "increasing": {"color": "#ef4444"}},
                    gauge={
                        "axis": {"range": [0, 35000], "tickcolor": "#475569", "tickfont": {"color": "#64748b"}},
                        "bar": {"color": tier_col, "thickness": 0.28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 5000], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [5000, 8500], "color": "rgba(2, 132, 199, 0.15)"},
                            {"range": [8500, 13000], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [13000, 20000], "color": "rgba(239, 68, 68, 0.15)"},
                            {"range": [20000, 35000], "color": "rgba(139, 92, 246, 0.2)"},
                        ],
                        "threshold": {
                            "line": {"color": tier_col, "width": 4},
                            "thickness": 0.8,
                            "value": predicted_inr,
                        },
                    }
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#f8fafc"},
                    height=290,
                    margin=dict(l=15, r=15, t=40, b=0),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#e2e8f0; margin-bottom:6px;'>Domestic Fare Tiers:</div>", unsafe_allow_html=True)
                scale_items = [
                    ("Super Saver", "< ₹5,000", "#10b981"),
                    ("Economy Standard", "₹5,000 – ₹8,500", "#0284c7"),
                    ("Flexible Economy", "₹8,500 – ₹13,000", "#f59e0b"),
                    ("Premium Fare", "₹13,000 – ₹20,000", "#ef4444"),
                    ("Luxury / Business", "> ₹20,000", "#8b5cf6"),
                ]
                for sc_cat, sc_range, sc_col in scale_items:
                    is_active = (sc_cat in tier)
                    active_border = f"border:1px solid {sc_col}; background:{sc_col}15;" if is_active else ""
                    pin = f"<span style='color:{sc_col}; font-weight:800;'>◀ CURRENT FLIGHT</span>" if is_active else ""
                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:5px 10px; border-radius:8px; margin:3px 0; font-size:0.82rem; {active_border}">
                      <div style="display:flex; align-items:center; gap:8px;">
                        <span style="width:10px; height:10px; border-radius:50%; background:{sc_col};"></span>
                        <span style="color:#f8fafc; font-weight:600;">{sc_cat}</span>
                        <span style="color:#64748b; font-size:0.78rem;">({sc_range})</span>
                      </div>
                      <div>{pin}</div>
                    </div>
                    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if df is None:
        st.error("Dataset not found. Make sure `dataset/Data_Train.csv` or `Data_Train.xlsx` exists.")
    else:
        st.markdown("<div class='section-header'>📂 Domestic Flights Dataset Explorer</div>", unsafe_allow_html=True)

        f1, f2, f3 = st.columns(3)
        with f1:
            sel_airlines = st.multiselect("Filter Airlines:", options=sorted(df['Airline'].dropna().unique()), default=[])
        with f2:
            sel_sources = st.multiselect("Filter Source City:", options=sorted(df['Source'].dropna().unique()), default=[])
        with f3:
            sel_stops = st.multiselect("Filter Total Stops:", options=sorted(df['Total_Stops'].dropna().unique()), default=[])

        filtered_df = df.copy()
        if sel_airlines:
            filtered_df = filtered_df[filtered_df['Airline'].isin(sel_airlines)]
        if sel_sources:
            filtered_df = filtered_df[filtered_df['Source'].isin(sel_sources)]
        if sel_stops:
            filtered_df = filtered_df[filtered_df['Total_Stops'].isin(sel_stops)]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Displaying Flights", f"{len(filtered_df):,}", f"{len(filtered_df) - len(df):,} filtered" if len(filtered_df) != len(df) else "Full Dataset")
        m2.metric("Airlines Tracked", str(filtered_df['Airline'].nunique()), "Operators")
        m3.metric("Average Fare", f"₹{filtered_df['Price'].mean():,.0f}", f"₹{filtered_df['Price'].mean() - 9087:,.0f} vs National")
        m4.metric("Fare Range", f"₹{filtered_df['Price'].min():,} – ₹{filtered_df['Price'].max():,}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(filtered_df.head(15), use_container_width=True, height=280)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📈 Exploratory Airfare Analytics</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.histogram(
                filtered_df, x='Price', nbins=40,
                title="Airfare Price Distribution",
                color_discrete_sequence=['#0284c7'],
                template='plotly_dark'
            )
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            airline_fares = (filtered_df.groupby('Airline')['Price'].mean().reset_index()
                             .sort_values('Price', ascending=False).head(10))
            fig2 = px.bar(
                airline_fares, x='Airline', y='Price',
                title="Top Airlines by Mean Ticket Price",
                color='Price', color_continuous_scale='Blues',
                template='plotly_dark'
            )
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            stops_fares = filtered_df.groupby('Total_Stops')['Price'].mean().reset_index()
            fig3 = px.bar(
                stops_fares, x='Total_Stops', y='Price',
                title="Average Fare by Total Stops",
                color='Price', color_continuous_scale='Teal',
                template='plotly_dark'
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            source_dest = pd.crosstab(filtered_df['Source'], filtered_df['Destination'], values=filtered_df['Price'], aggfunc='mean').fillna(0)
            fig4 = px.imshow(
                source_dest, text_auto=",.0f",
                title="Source vs Destination Mean Fare Matrix (₹)",
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8')
            st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🤖 10-Fold Cross-Validation Leaderboard</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(56, 189, 248, 0.05); border:1px solid rgba(56, 189, 248, 0.2);
                border-radius:12px; padding:0.9rem 1.2rem; margin-bottom:1.2rem; font-size:0.88rem; color:#94a3b8;">
      💡 <strong>Rigorous 10-Fold Cross-Validation:</strong> Evaluated across 10,682 records with zero data leakage. 5 distinct regression models compared using Root Mean Squared Error (RMSE).
    </div>
    """, unsafe_allow_html=True)

    # Top 3 Podium
    pod1, pod2, pod3 = st.columns(3)
    with pod1:
        st.markdown("""
        <div class="metric-card" style="border-color:#38bdf8;">
          <div style="font-size:2rem;">🥇</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.1rem; margin-top:4px;">Random Forest Regressor</div>
          <div style="color:#38bdf8; font-weight:800; font-size:1.6rem; font-family:'Space Grotesk'; margin:4px 0;">RMSE: ₹2,051</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.8011 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod2:
        st.markdown("""
        <div class="metric-card" style="border-color:#818cf8;">
          <div style="font-size:2rem;">🥈</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.1rem; margin-top:4px;">Extra Trees Regressor</div>
          <div style="color:#818cf8; font-weight:800; font-size:1.6rem; font-family:'Space Grotesk'; margin:4px 0;">RMSE: ₹2,251</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.7601 · 100 Estimators</div>
        </div>
        """, unsafe_allow_html=True)
    with pod3:
        st.markdown("""
        <div class="metric-card" style="border-color:#10b981;">
          <div style="font-size:2rem;">🥉</div>
          <div style="font-weight:700; color:#f8fafc; font-size:1.1rem; margin-top:4px;">SVR (RBF Kernel)</div>
          <div style="color:#10b981; font-weight:800; font-size:1.6rem; font-family:'Space Grotesk'; margin:4px 0;">RMSE: ₹3,155</div>
          <div style="font-size:0.75rem; color:#64748b;">R²: 0.5314 · Scaled C=100</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Leaderboard Table
    algo_df = pd.DataFrame({
        "Rank": ["🥇 1st", "🥈 2nd", "🥉 3rd", "4th", "5th"],
        "Algorithm": ["Random Forest Regressor", "Extra Trees Regressor", "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "Category": ["Bagged Decision Trees", "Extremely Randomized Trees", "Support Vector Machine", "L2 Regularized Linear", "L1 Regularized Linear"],
        "10-Fold CV RMSE": ["₹2,050.99", "₹2,250.72", "₹3,154.68", "₹3,445.00", "₹3,445.00"],
        "R² Score": ["0.8011", "0.7601", "0.5314", "0.4408", "0.4408"],
        "Key Configuration": [
            "n_estimators=100, max_features='sqrt'",
            "n_estimators=100, max_features='sqrt'",
            "C=100, gamma='scale', RBF",
            "alpha=1.0, StandardScaler",
            "alpha=0.1, max_iter=10000",
        ]
    })
    st.dataframe(algo_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 10-Fold CV Performance Visualizer</div>", unsafe_allow_html=True)

    sim_df = pd.DataFrame({
        "Model": ["Random Forest Regressor", "Extra Trees Regressor", "SVR (RBF Kernel)", "Ridge Regression", "Lasso Regression"],
        "RMSE (₹)": [2050.99, 2250.72, 3154.68, 3445.00, 3445.00],
        "Color": ["#38bdf8", "#818cf8", "#10b981", "#f59e0b", "#ef4444"],
    })

    fig_rmse = go.Figure()
    fig_rmse.add_trace(go.Bar(
        y=sim_df["Model"],
        x=sim_df["RMSE (₹)"],
        orientation='h',
        marker_color=sim_df["Color"],
        text=[f"₹{v:,.0f}" for v in sim_df["RMSE (₹)"]],
        textposition='outside',
    ))
    fig_rmse.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis_title="10-Fold Cross-Validation RMSE (₹) — Lower is Better",
        yaxis=dict(autorange='reversed'),
        height=320,
        margin=dict(l=10, r=80, t=10, b=40),
    )
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔬 Cross-Validation Methodology</div>", unsafe_allow_html=True)

    cv_c1, cv_c2 = st.columns(2)
    with cv_c1:
        st.markdown("""
        <div class="glass-panel">
          <div style="font-weight:700; color:#38bdf8; margin-bottom:6px;">10-Fold Partitioning Protocol</div>
          <p style="font-size:0.86rem; color:#94a3b8; line-height:1.6;">
            1. Entire flight dataset (10,682 records) shuffled with <code>random_state=42</code>.<br>
            2. Split into 10 equal disjoint partitions.<br>
            3. Iteratively 9 folds (~9,614 flights) train the regressor while 1 fold (~1,068 flights) evaluates out-of-sample accuracy.<br>
            4. Guarantees true generalization and safeguards against test data leakage.
          </p>
        </div>
        """, unsafe_allow_html=True)
    with cv_c2:
        st.markdown("""
        <div class="glass-panel">
          <div style="font-weight:700; color:#818cf8; margin-bottom:6px;">Root Mean Squared Error (RMSE)</div>
          <p style="font-size:0.86rem; color:#94a3b8; line-height:1.6;">
            Penalizes large pricing discrepancies quadratically in native currency units (₹):<br>
            <code style="color:#f8fafc; font-size:0.85rem;">RMSE = √( 1/n Σ(Price_i - ŷ_i)² )</code><br>
            Tree ensembles vastly outperform linear models by capturing non-linear interactions between routes, departure time windows, and advance booking days.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.8rem; padding: 1rem 0 1.5rem;">
  <strong>AIC354 Machine Learning Fundamentals · Assignment 3</strong><br>
  Flight Fare Prediction & Cloud Deployment AI System<br>
  Dataset: 10,683 Domestic Flights (India)
</div>
""", unsafe_allow_html=True)
