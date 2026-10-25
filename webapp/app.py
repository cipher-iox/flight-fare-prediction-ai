"""
Flight Fare Prediction — Flask Web Application
AIC354 Machine Learning Fundamentals | Assignment 3
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model bundle ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

model_bundle = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_bundle = pickle.load(f)
        print(f"✅ Model loaded: {model_bundle.get('model_name', 'Trained Model')}")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
else:
    print(f"⚠️ Model not found at {MODEL_PATH}. Run the notebook first.")

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

STOPS_OPTIONS = [
    {'label': 'Non-stop (Direct)', 'value': 0},
    {'label': '1 Stop', 'value': 1},
    {'label': '2 Stops', 'value': 2},
    {'label': '3 Stops', 'value': 3},
    {'label': '4 Stops', 'value': 4},
]


@app.route('/')
def index():
    model_name = model_bundle.get('model_name', 'Model Offline') if model_bundle else 'Model Offline'
    return render_template(
        'index.html',
        airlines=AIRLINES,
        sources=SOURCES,
        destinations=DESTINATIONS,
        stops=STOPS_OPTIONS,
        model_name=model_name
    )


@app.route('/predict', methods=['POST'])
def predict():
    if model_bundle is None:
        return jsonify({
            'error': 'Model not loaded. Train and save best_model.pkl to the models/ directory.'
        }), 500

    try:
        data = request.get_json() or {}

        airline = data.get('airline', 'IndiGo')
        source = data.get('source', 'Banglore')
        destination = data.get('destination', 'Delhi')
        total_stops = int(data.get('total_stops', 0))

        # Date parsing
        journey_day = int(data.get('journey_day', 15))
        journey_month = int(data.get('journey_month', 5))
        journey_dayofweek = int(data.get('journey_dayofweek', 2))

        # Time parsing
        dep_hour = int(data.get('dep_hour', 10))
        dep_min = int(data.get('dep_min', 30))
        arrival_hour = int(data.get('arrival_hour', 13))
        arrival_min = int(data.get('arrival_min', 15))

        # Duration
        duration_mins = int(data.get('duration_mins', 165))

        # Safe encoding
        le_dict = model_bundle.get('le_dict', {})
        features = model_bundle.get('features', [])
        model = model_bundle.get('model')

        def safe_encode(le, val):
            if le is not None and hasattr(le, 'classes_') and val in le.classes_:
                return int(le.transform([val])[0])
            return 0

        airline_enc = safe_encode(le_dict.get('Airline'), airline)
        source_enc = safe_encode(le_dict.get('Source'), source)
        dest_enc = safe_encode(le_dict.get('Destination'), destination)

        row_dict = {
            'Airline': airline_enc,
            'Source': source_enc,
            'Destination': dest_enc,
            'Total_Stops': total_stops,
            'Journey_Day': journey_day,
            'Journey_Month': journey_month,
            'Journey_DayOfWeek': journey_dayofweek,
            'Dep_Hour': dep_hour,
            'Dep_Min': dep_min,
            'Arrival_Hour': arrival_hour,
            'Arrival_Min': arrival_min,
            'Duration_Mins': duration_mins,
        }

        if features:
            input_df = pd.DataFrame([row_dict])[features]
            raw_pred = float(model.predict(input_df)[0])
        else:
            input_arr = np.array([list(row_dict.values())])
            raw_pred = float(model.predict(input_arr)[0])

        INR_TO_PKR = 3.32
        fare_pkr = max(5000, round(raw_pred * INR_TO_PKR))

        # Tier calculation in PKR
        if fare_pkr < 16500:
            tier, color = "Saver Economy", "#10b981"
        elif fare_pkr < 28000:
            tier, color = "Standard Economy", "#3b82f6"
        elif fare_pkr < 45000:
            tier, color = "Flexible Economy", "#f59e0b"
        elif fare_pkr < 70000:
            tier, color = "Premium Fare", "#ef4444"
        else:
            tier, color = "Business / Luxury", "#a855f7"

        return jsonify({
            'fare': fare_pkr,
            'fare_formatted': f"Rs. {fare_pkr:,}",
            'fare_usd': round(fare_pkr / 278.5, 2),
            'tier': tier,
            'color': color,
            'airline': airline,
            'route': f"{source} ➔ {destination}",
            'model_used': model_bundle.get('model_name', 'Random Forest Regressor'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Guard against accidental execution by Streamlit Community Cloud runner
    import sys
    if any('streamlit' in arg.lower() for arg in sys.argv) or 'streamlit' in sys.modules:
        import streamlit as st
        st.set_page_config(page_title="Configuration Notice", page_icon="⚙️")
        st.error("⚠️ **Incorrect Main File Path Detected on Streamlit Cloud**")
        st.warning("""
        You configured the Streamlit Cloud entrypoint to `webapp/app.py` (which is the Flask web server).
        
        ### 🛠️ How to fix in 10 seconds:
        1. Click the **Manage app** menu (the 3 dots `⋮` at the bottom right).
        2. Go to **Settings** ➔ **General**.
        3. Change **Main file path** from `webapp/app.py` to **`app.py`**.
        4. Click **Save** — your full Streamlit UI will load instantly!
        """)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
