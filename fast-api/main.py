import os
import warnings
from typing import Literal, List

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import dropbox
from dropbox.exceptions import AuthError
from sklearn.exceptions import InconsistentVersionWarning

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

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
# 1. FastAPI app
# =========================
app = FastAPI(title="Rain Prediction API")

# Global artifacts
MODELS = {}
feature_columns: List[str] = []
scaler: StandardScaler | None = None

# Columns to be scaled (must match training)
SCALE_COLS = [
    "Pressure9am", "Pressure3pm",
    "Humidity9am", "Humidity3pm",
    "Temp9am", "Temp3pm",
    "WindSpeed9am", "WindSpeed3pm",
]


# =========================
# 2. Environment and Dropbox
# =========================
load_dotenv()

DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")
DROPBOX_FOLDER = os.getenv("DROPBOX_FOLDER", "/weather-ml-upload")

if not all([DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN]):
    raise RuntimeError(
        "Dropbox credentials not found in environment variables. "
        "Expected DROPBOX_APP_KEY, DROPBOX_APP_SECRET, DROPBOX_REFRESH_TOKEN."
    )

try:
    dbx = dropbox.Dropbox(
        oauth2_access_token=None,
        oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
        app_key=DROPBOX_APP_KEY,
        app_secret=DROPBOX_APP_SECRET,
    )
except AuthError as e:
    raise RuntimeError(f"Failed to initialize Dropbox client: {e}")


os.makedirs("artifacts", exist_ok=True)


def download_latest(prefix: str, local_path: str) -> None:
    """Download latest file from Dropbox folder whose name starts with given prefix."""
    print(f"[INFO] Looking for latest file with prefix: {prefix}")

    try:
        result = dbx.files_list_folder(DROPBOX_FOLDER)
    except AuthError as e:
        raise RuntimeError(f"Dropbox auth error while listing folder: {e}")

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

    # Stream file directly to disk to avoid high memory usage
    dbx.files_download_to_file(local_path, latest_file.path_lower)

    print(f"[INFO] Saved to: {local_path}")


# =========================
# 3. Pydantic models
# =========================
LocationType = Literal["Sydney", "Melbourne", "Brisbane", "Perth"]
WindDirType = Literal["N", "S", "E", "W", "NE", "NW", "SE", "SW"]
ModelNameType = Literal["Logistic Regression", "Random Forest", "XGBoost"]


class PredictRequest(BaseModel):
    model_name: ModelNameType = Field(
        default="Random Forest",
        description="Which model to use for prediction",
    )

    Location: LocationType
    WindGustDir: WindDirType
    WindDir9am: WindDirType
    WindDir3pm: WindDirType

    MinTemp: float
    MaxTemp: float
    Rainfall: float
    WindGustSpeed: float

    Humidity9am: int = Field(ge=0, le=100)
    Humidity3pm: int = Field(ge=0, le=100)

    Pressure9am: float
    Pressure3pm: float

    Temp9am: float
    Temp3pm: float

    WindSpeed9am: float
    WindSpeed3pm: float

    RainToday: bool

    Year: int
    Month: int = Field(ge=1, le=12)
    Day: int = Field(ge=1, le=31)


class PredictResponse(BaseModel):
    model_name: str
    will_rain: bool
    probability: float


class ModelsResponse(BaseModel):
    models: List[str]


# =========================
# 4. Startup: load artifacts
# =========================
@app.on_event("startup")
def load_models_and_artifacts() -> None:
    """Download models and preprocessing artifacts from Dropbox and load them into memory."""
    global MODELS, feature_columns, scaler

    print("[INFO] Loading models and artifacts from Dropbox...")

    download_latest("logreg_model", "artifacts/logreg_model.joblib")
    download_latest("rf_model", "artifacts/rf_model.joblib")
    download_latest("xgb_model", "artifacts/xgb_model.joblib")
    download_latest("feature_columns", "artifacts/feature_columns.joblib")
    download_latest("scaler", "artifacts/scaler.joblib")

    logreg_model: LogisticRegression = joblib.load("artifacts/logreg_model.joblib")
    rf_model: RandomForestClassifier = joblib.load("artifacts/rf_model.joblib")
    xgb_model: XGBClassifier = joblib.load("artifacts/xgb_model.joblib")

    feature_cols = joblib.load("artifacts/feature_columns.joblib")
    loaded_scaler: StandardScaler = joblib.load("artifacts/scaler.joblib")

    MODELS = {
        "Logistic Regression": logreg_model,
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
    }
    feature_columns[:] = list(feature_cols)
    # type: ignore
    globals()["scaler"] = loaded_scaler

    print("[INFO] All models and artifacts loaded successfully")
    print("[INFO] Available models:", list(MODELS.keys()))
    print("[INFO] Feature columns:", len(feature_columns))


# =========================
# 5. Helper: preprocessing
# =========================
def build_input_dataframe(req: PredictRequest) -> pd.DataFrame:
    """Build and preprocess input DataFrame from request."""
    if not feature_columns or scaler is None:
        raise RuntimeError("Models are not loaded yet")

    # Build initial single-row DataFrame
    df = pd.DataFrame([{
        "Location": req.Location,
        "WindGustDir": req.WindGustDir,
        "WindDir9am": req.WindDir9am,
        "WindDir3pm": req.WindDir3pm,
        "MinTemp": req.MinTemp,
        "MaxTemp": req.MaxTemp,
        "Rainfall": req.Rainfall,
        "WindGustSpeed": req.WindGustSpeed,
        "Humidity9am": req.Humidity9am,
        "Humidity3pm": req.Humidity3pm,
        "Pressure9am": req.Pressure9am,
        "Pressure3pm": req.Pressure3pm,
        "Temp9am": req.Temp9am,
        "Temp3pm": req.Temp3pm,
        "WindSpeed9am": req.WindSpeed9am,
        "WindSpeed3pm": req.WindSpeed3pm,
        "RainToday": 1 if req.RainToday else 0,
        "Year": req.Year,
        "Month": req.Month,
        "Day": req.Day,
    }])

    # Same preprocessing as in Streamlit app
    df["Rainfall"] = np.log1p(df["Rainfall"])
    df["WindGustSpeed"] = np.log1p(df["WindGustSpeed"])

    df = pd.get_dummies(
        df,
        columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"],
        drop_first=True,
    )

    df = df.reindex(columns=feature_columns, fill_value=0)

    df[SCALE_COLS] = scaler.transform(df[SCALE_COLS])

    return df


# =========================
# 6. API endpoints
# =========================
@app.get("/models", response_model=ModelsResponse)
def list_models():
    """Return list of available models."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")
    return ModelsResponse(models=list(MODELS.keys()))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Make prediction using selected model."""
    if not MODELS:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    if req.model_name not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_name: {req.model_name}",
        )

    try:
        df = build_input_dataframe(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")

    model = MODELS[req.model_name]

    try:
        pred = model.predict(df)[0]
        proba = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    return PredictResponse(
        model_name=req.model_name,
        will_rain=bool(pred == 1),
        probability=proba,
    )

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

