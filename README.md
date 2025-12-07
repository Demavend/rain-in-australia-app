# Rain in Australia – ML Model + API + Docker

This repository contains a small end-to-end machine learning project:

- training several models to predict **whether it will rain tomorrow in Australia**
- an interactive **Streamlit** app for manual exploration
- a production-oriented **FastAPI** service that exposes a simple HTTP API
- a **Docker** image for the FastAPI service (suitable for deployment on Render)

The project is used as a homework example: *“Create a simple API for your ML model and deploy it in a Docker container.”*

---

## 1. Data and Problem

- **Dataset:** Rain in Australia (Kaggle, stored locally as `assets/weatherAUS.csv.zip`)
- **Target:** `RainTomorrow` – whether it rained the next day (binary classification)
- **Typical features:**
  - Location, Date (Year / Month / Day)
  - Temperature (min / max / at 9am / at 3pm)
  - Pressure, humidity, wind speed & direction, rainfall, etc.

**Goal:** given today’s weather conditions, predict the probability that it will rain tomorrow.

---

## 2. Models and Training

Model training and feature engineering live in:

- `train.ipynb`

During training the code:

1. Loads and cleans the weather dataset.
2. Performs basic feature engineering and preprocessing:
   - log-transform for some numeric variables (e.g. rainfall)
   - one-hot encoding for categorical features
   - standard scaling for selected numeric columns.
3. Trains several models:
   - **Logistic Regression**
   - **Random Forest**
   - **XGBoost**
4. Saves the following artifacts using `joblib`:
   - `logreg_model_*.joblib`
   - `rf_model_*.joblib`
   - `xgb_model_*.joblib`
   - `feature_columns_*.joblib`
   - `scaler_*.joblib`

These artifacts are uploaded to Dropbox and later used by the FastAPI service.

**Main technologies:**

- Python 3.12
- `pandas`, `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

---

## 3. Repository Structure

```
rain-in-australia-app/
├── assets/
│   └── weatherAUS.csv.zip
├── artifacts/
├── fast-api/
│   ├── artifacts/
│   ├── static/
│   │   ├── index.html
│   │   └── app.js
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── app.py
├── Dockerfile
├── get_refresh_token.py
├── requirements.txt
├── train.ipynb
└── README.md
```

The part that strictly satisfies the homework requirement “simple API + Docker” is everything inside the `fast-api/` directory.

---

## 4. FastAPI Inference Service

On startup the FastAPI application:

- Reads Dropbox credentials from environment variables:
  - `DROPBOX_APP_KEY`
  - `DROPBOX_APP_SECRET`
  - `DROPBOX_REFRESH_TOKEN`
  - `DROPBOX_FOLDER` (default: `/weather-ml-upload`)
- Downloads the latest model artifacts from the configured Dropbox folder
- Loads the models, feature list and scaler into memory

### Endpoints

**GET /models**  
Returns the list of available models.

**POST /predict**  
Accepts a JSON payload describing today’s weather and returns the prediction and probability.

---

## 4.2 Local Development (without Docker)

From the `fast-api/` folder:

```
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open:
- UI: http://127.0.0.1:8000/
- Docs: http://127.0.0.1:8000/docs

Make sure the Dropbox variables are available in `.env` or your shell.

---

## 5. Docker Image for the API

Dockerfile is located in `fast-api/Dockerfile`.

Build and run locally:

```
docker build -t rain-api .
docker run -e PORT=8000 -e DROPBOX_APP_KEY=... -e DROPBOX_APP_SECRET=... -e DROPBOX_REFRESH_TOKEN=... -e DROPBOX_FOLDER=/weather-ml-upload -p 8000:8000 rain-api
```

---

## 6. Frontend (Bootstrap UI)

Implemented in:

- `fast-api/static/index.html`
- `fast-api/static/app.js`

Features:
- Button-based selectors for model and location
- Sliders for humidity
- HTML5 date picker for observation date

---

## 7. Streamlit Demo (Legacy)

The repository still contains the original Streamlit demo:

- `app.py`
- root-level `Dockerfile`
- root-level `requirements.txt`

This version is kept only for comparison.

---

## 8. Deployed Endpoint

Endpoint URL:
https://YOUR-RENDER-APP-NAME.onrender.com/

UI:
https://YOUR-RENDER-APP-NAME.onrender.com/

API Docs:
https://YOUR-RENDER-APP-NAME.onrender.com/docs

Prediction Endpoint:
POST https://YOUR-RENDER-APP-NAME.onrender.com/predict

---

## 9. Homework Submission Summary

- Repository with code – this GitHub repository
- Short project description – sections 1–3
- Link to a working endpoint – section 8
- API deployed in Docker container using Render
