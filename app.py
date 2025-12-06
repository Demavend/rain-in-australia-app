import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import dropbox

from dotenv import load_dotenv
from sklearn.exceptions import InconsistentVersionWarning


# =========================
# 0. Warnings filtering
# =========================
# Ignore sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Ignore xgboost serialized-model warning text
warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*xgboost.*",
)


# =========================
# 1. Load environment
# =========================
load_dotenv()

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
DROPBOX_FOLDER = os.getenv("DROPBOX_FOLDER", "/weather-ml-upload")

if DROPBOX_ACCESS_TOKEN is None:
    st.error("Dropbox token not found in environment variables")
    st.stop()

dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)


# =========================
# 2. Download latest artifacts from Dropbox by prefix
# =========================
os.makedirs("artifacts", exist_ok=True)


def download_latest(prefix: str, local_path: str) -> None:
    """Download latest file from Dropbox folder whose name starts with given prefix."""
    print(f"[INFO] Looking for latest file with prefix: {prefix}")

    result = dbx.files_list_folder(DROPBOX_FOLDER)

    candidates = [
        entry
        for entry in result.entries
        if isinstance(entry, dropbox.files.FileMetadata)
        and entry.name.startswith(prefix)
    ]

    if not candidates:
        raise RuntimeError(f"No file found in Dropbox with prefix: {prefix}")

    latest_file = sorted(
        candidates,
        key=lambda x: x.client_modified
    )[-1]

    print(f"[INFO] Downloading: {latest_file.name}")

    md, res = dbx.files_download(latest_file.path_lower)
    with open(local_path, "wb") as f:
        f.write(res.content)

    print(f"[INFO] Saved to: {local_path}")


@st.cache_resource
def load_models_and_artifacts():
    """
    Download models and preprocessing artifacts from Dropbox (once per session)
    and load them into memory.
    """
    print("[INFO] Loading models and artifacts from Dropbox...")

    download_latest("logreg_model",    "artifacts/logreg_model.joblib")
    download_latest("rf_model",        "artifacts/rf_model.joblib")
    download_latest("xgb_model",       "artifacts/xgb_model.joblib")
    download_latest("feature_columns", "artifacts/feature_columns.joblib")
    download_latest("scaler",          "artifacts/scaler.joblib")

    logreg_model = joblib.load("artifacts/logreg_model.joblib")
    rf_model     = joblib.load("artifacts/rf_model.joblib")
    xgb_model    = joblib.load("artifacts/xgb_model.joblib")

    feature_columns = joblib.load("artifacts/feature_columns.joblib")
    scaler = joblib.load("artifacts/scaler.joblib")

    models = {
        "Logistic Regression": logreg_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model
    }

    print("[INFO] All models and artifacts loaded successfully")

    return models, feature_columns, scaler


MODELS, feature_columns, scaler = load_models_and_artifacts()

# Default selected model in segmented control
if "model_name" not in st.session_state:
    st.session_state["model_name"] = "Random Forest"


# =========================
# 4. Streamlit UI
# =========================
st.title("Will it rain tomorrow?")
st.write(
    "Fill in today's weather conditions to predict whether it will rain tomorrow "
    "and see the estimated probability."
)

model_name = st.segmented_control(
    "Choose prediction model:",
    list(MODELS.keys()),
    key="model_name"
)

st.subheader("Weather data for today")


Location = st.selectbox(
    "City (Location):",
    ["Sydney", "Melbourne", "Brisbane", "Perth"]
)

WindGustDir = st.selectbox(
    "Wind gust direction (max gust):",
    ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
)

WindDir9am = st.selectbox(
    "Wind direction at 9:00:",
    ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
)

WindDir3pm = st.selectbox(
    "Wind direction at 15:00:",
    ["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
)

MinTemp = st.number_input(
    "Minimum temperature today (°C):",
    value=10.0
)

MaxTemp = st.number_input(
    "Maximum temperature today (°C):",
    value=20.0
)

Rainfall = st.number_input(
    "Rainfall today (mm):",
    value=0.0
)

WindGustSpeed = st.number_input(
    "Maximum wind gust speed (km/h):",
    value=30.0
)

Humidity9am = st.slider(
    "Humidity at 9:00 (%):",
    0,
    100,
    60
)

Humidity3pm = st.slider(
    "Humidity at 15:00 (%):",
    0,
    100,
    50
)

Pressure9am = st.number_input(
    "Air pressure at 9:00 (hPa):",
    value=1012.0
)

Pressure3pm = st.number_input(
    "Air pressure at 15:00 (hPa):",
    value=1010.0
)

Temp9am = st.number_input(
    "Temperature at 9:00 (°C):",
    value=15.0
)

Temp3pm = st.number_input(
    "Temperature at 15:00 (°C):",
    value=22.0
)

WindSpeed9am = st.number_input(
    "Wind speed at 9:00 (km/h):",
    value=10.0
)

WindSpeed3pm = st.number_input(
    "Wind speed at 15:00 (km/h):",
    value=15.0
)

RainToday = st.selectbox(
    "Did it rain today?",
    ["No", "Yes"]
)

Year = st.number_input(
    "Observation year:",
    value=2024
)

Month = st.number_input(
    "Observation month:",
    value=6,
    min_value=1,
    max_value=12
)

Day = st.number_input(
    "Observation day:",
    value=15,
    min_value=1,
    max_value=31
)


# =========================
# 5. Build input dataframe
# =========================
input_data = pd.DataFrame([{
    "Location": Location,
    "WindGustDir": WindGustDir,
    "WindDir9am": WindDir9am,
    "WindDir3pm": WindDir3pm,
    "MinTemp": MinTemp,
    "MaxTemp": MaxTemp,
    "Rainfall": Rainfall,
    "WindGustSpeed": WindGustSpeed,
    "Humidity9am": Humidity9am,
    "Humidity3pm": Humidity3pm,
    "Pressure9am": Pressure9am,
    "Pressure3pm": Pressure3pm,
    "Temp9am": Temp9am,
    "Temp3pm": Temp3pm,
    "WindSpeed9am": WindSpeed9am,
    "WindSpeed3pm": WindSpeed3pm,
    "RainToday": 1 if RainToday == "Yes" else 0,
    "Year": Year,
    "Month": Month,
    "Day": Day
}])


# =========================
# 6. Preprocessing
# =========================
input_data["Rainfall"] = np.log1p(input_data["Rainfall"])
input_data["WindGustSpeed"] = np.log1p(input_data["WindGustSpeed"])

input_data = pd.get_dummies(
    input_data,
    columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"],
    drop_first=True
)

input_data = input_data.reindex(columns=feature_columns, fill_value=0)

scale_cols = [
    "Pressure9am", "Pressure3pm",
    "Humidity9am", "Humidity3pm",
    "Temp9am", "Temp3pm",
    "WindSpeed9am", "WindSpeed3pm"
]

input_data[scale_cols] = scaler.transform(input_data[scale_cols])


# =========================
# 7. Prediction
# =========================
if st.button("Get prediction"):
    model = MODELS[model_name]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    will_rain = (prediction == 1)

    st.subheader("Prediction result")
    st.write(f"**Will it rain tomorrow?** {'Yes' if will_rain else 'No'}")
    st.write(f"**Estimated probability of rain:** {probability:.2%}")
