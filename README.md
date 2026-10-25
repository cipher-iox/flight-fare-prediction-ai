# ✈️ Flight Fare Prediction — AI Airfare Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)](https://xgboost.readthedocs.io)
[![Google Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AIC354 – Machine Learning Fundamentals | Assignment 3**  
> Applying, Evaluating, and Comparing Multiple ML Algorithms on a Large Real-World Dataset.

---

## 📌 Problem Statement

Predict commercial **flight ticket fares (INR ₹)** across major domestic Indian flight corridors based on itinerary parameters (Airline carrier, Source airport, Destination, Stops, Duration, and Departure/Arrival schedules). Five distinct machine learning algorithms are rigorously evaluated and compared using **10-Fold Cross-Validation** on the full 10,683-flight dataset to avoid overfitting and data leakage.

---

## 📂 Repository Structure

```
flight-fare-prediction-ai/
├── dataset/
│   ├── Data_Train.csv                  # Full dataset (10,683 records in CSV)
│   └── Data_Train.xlsx                 # Original raw Excel format
├── notebooks/
│   └── flight_fare_prediction_full_10fold.ipynb   # Master Notebook: 10-Fold CV & EDA
├── models/
│   ├── README.md                       # Model directory documentation
│   └── best_model.pkl                  # Pickled top-performing trained model bundle
├── webapp/                             # Standalone Flask Web Application
│   ├── app.py                          # Flask REST API & Controller
│   ├── requirements.txt                # Flask dependencies
│   ├── templates/
│   │   └── index.html                  # Responsive modern frontend UI
│   └── static/
│       └── style.css                   # Glassmorphic dark-theme styling
├── app.py                              # Streamlit Web Application (Community Cloud ready)
├── requirements.txt                    # Python dependencies for Streamlit
├── .gitignore                          # Git ignore definitions
└── README.md                           # Documentation & Deployment Guide
```

---

## 🤖 5 Machine Learning Algorithms Compared

All algorithms are evaluated using **10-Fold Cross-Validation** with **RMSE** (Root Mean Squared Error) and **R² Score**:

| # | Algorithm | Category | Key Configuration |
|---|-----------|----------|-------------------|
| 1 | **Ridge Regression** | Regularized Linear (L2) | `alpha=1.0`, `StandardScaler` Pipeline |
| 2 | **Lasso Regression** | Regularized Linear (L1) | `alpha=0.1`, `max_iter=10000`, `StandardScaler` Pipeline |
| 3 | **Support Vector Regression (SVR)** | Kernel-Based | `kernel='rbf'`, `C=100`, `gamma='scale'`, `StandardScaler` Pipeline |
| 4 | **Extra Trees Regressor** | Ensemble (Bagging) | `n_estimators=100`, `max_features='sqrt'`, `random_state=42` |
| 5 | **Random Forest / XGBoost Regressor** | Ensemble (Boosting/Trees) | `n_estimators=100`, `max_features='sqrt'`, `random_state=42` |

---

## 📊 Dataset Summary

| Property | Value |
|----------|-------|
| **Records** | 10,683 flight instances |
| **Features** | 11 attributes (Airline, Source, Destination, Stops, Duration, Dep/Arr Time, Date) |
| **Target Variable** | `Price` (Ticket Fare in INR ₹) |
| **Price Range** | ₹1,759 – ₹79,512 (Mean: ~₹9,087) |
| **Major Airlines** | Jet Airways, IndiGo, Air India, Multiple carriers, SpiceJet, Vistara, GoAir, Air Asia |
| **Major Hubs** | Delhi, Kolkata, Banglore, Mumbai, Chennai, Cochin, Hyderabad |

---

## 🚀 How to Run on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File -> Upload notebook** and select `notebooks/flight_fare_prediction_full_10fold.ipynb`.
3. Click **Runtime -> Run all**.
4. When prompted, upload `dataset/Data_Train.csv` (or `Data_Train.xlsx`).
5. The notebook will automatically:
   - Perform full Exploratory Data Analysis (EDA) and save visual plots.
   - Execute 10-Fold Cross-Validation across all 5 models.
   - Generate RMSE and R² leaderboard comparison tables and boxplots.
   - Fit the winning model on all data and download `best_model.pkl`.

---

## 🌐 Deploy on Streamlit Community Cloud (Step-by-Step)

The Streamlit web application ([`app.py`](app.py)) is configured for 1-click cloud deployment:

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "feat: complete flight fare prediction AI system"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/flight-fare-prediction-ai.git
git push -u origin main
```

### 2. Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
2. Click **"Create app"** (or **"New app"**).
3. Select your repository: `flight-fare-prediction-ai`.
4. Set **Branch:** `main`.
5. Set **Main file path:** `app.py`.
6. Click **"Deploy"**! 🎉

### 3. Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Alternative Flask Deployment

To run the lightweight Flask server:
```bash
cd webapp
pip install -r requirements.txt
python app.py
```
Access the application at [http://localhost:5000](http://localhost:5000).

---

## 👤 Author & Academic Context

- **Course:** AIC354 – Machine Learning Fundamentals
- **Assignment:** Lab Assignment 3 (Model Comparison & Evaluation)
- **Methodology:** 10-Fold Cross-Validation without data leakage
- **Submission:** Full dataset coverage with single master notebook & interactive Streamlit app

---

## 📜 License
Educational project developed for AIC354 Machine Learning Fundamentals.
