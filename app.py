import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import dropbox

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from dropbox.files import WriteMode


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

def download_latest(prefix: str, local_path: str):
    # List all files in the target folder
    result = dbx.files_list_folder(DROPBOX_FOLDER)

    # Keep only files whose name starts with the given prefix
    candidates = [
        entry for entry in result.entries
        if isinstance(entry, dropbox.files.FileMetadata)
        and entry.name.startswith(prefix)
    ]

    if not candidates:
        st.error(f"No file found in Dropbox with prefix: {prefix}")
        st.stop()

    # Pick latest by modification time
    latest_file = sorted(
        candidates,
        key=lambda x: x.client_modified
    )[-1]

    md, res = dbx.files_download(latest_file.path_lower)
    with open(local_path, "wb") as f:
        f.write(res.content)

    st.write(f"Loaded {prefix} from Dropbox file: {latest_file.name}")


# Download all required artifacts
download_latest("logreg_model",       "artifacts/logreg_model.joblib")
download_latest("rf_model",           "artifacts/rf_model.joblib")
download_latest("xgb_model",          "artifacts/xgb_model.joblib")
download_latest("feature_columns",    "artifacts/feature_columns.joblib")
download_latest("scaler",             "artifacts/scaler.joblib")


# =========================
# 3. Load models and artifacts
# =========================
logreg_model = joblib.load("artifacts/logreg_model.joblib")
rf_model     = joblib.load("artifacts/rf_model.joblib")
xgb_model    = joblib.load("artifacts/xgb_model.joblib")

feature_columns = joblib.load("artifacts/feature_columns.joblib")
scaler = joblib.load("artifacts/scaler.joblib")

MODELS = {
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}


# =========================
# 4. Streamlit UI
# =========================
st.title("Weather Rain Prediction")
st.write("Enter weather parameters to predict whether it will rain tomorrow.")

model_name = st.selectbox(
    "Select model:",
    list(MODELS.keys()),
    index=1  # default: Random Forest
)

st.subheader("Weather input")

Location = st.selectbox("Location", ["Sydney", "Melbourne", "Brisbane", "Perth"])
WindGustDir = st.selectbox("WindGustDir", ["N", "S", "E", "W", "NE", "NW", "SE", "SW"])
WindDir9am = st.selectbox("WindDir9am", ["N", "S", "E", "W", "NE", "NW", "SE", "SW"])
WindDir3pm = st.selectbox("WindDir3pm", ["N", "S", "E", "W", "NE", "NW", "SE", "SW"])

MinTemp = st.number_input("MinTemp", value=10.0)
MaxTemp = st.number_input("MaxTemp", value=20.0)
Rainfall = st.number_input("Rainfall", value=0.0)
WindGustSpeed = st.number_input("WindGustSpeed", value=30.0)
Humidity9am = st.slider("Humidity9am", 0, 100, 60)
Humidity3pm = st.slider("Humidity3pm", 0, 100, 50)
Pressure9am = st.number_input("Pressure9am", value=1012.0)
Pressure3pm = st.number_input("Pressure3pm", value=1010.0)
Temp9am = st.number_input("Temp9am", value=15.0)
Temp3pm = st.number_input("Temp3pm", value=22.0)
WindSpeed9am = st.number_input("WindSpeed9am", value=10.0)
WindSpeed3pm = st.number_input("WindSpeed3pm", value=15.0)
RainToday = st.selectbox("RainToday", ["No", "Yes"])

Year = st.number_input("Year", value=2024)
Month = st.number_input("Month", value=6)
Day = st.number_input("Day", value=15)


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
if st.button("Predict"):
    model = MODELS[model_name]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Yes" if prediction == 1 else "No"

    st.subheader("Prediction result")
    st.write(f"**Rain tomorrow:** {result}")
    st.write(f"**Probability:** {probability:.2%}")
