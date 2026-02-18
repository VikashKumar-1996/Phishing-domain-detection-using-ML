from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

from pathlib import Path
from pydantic import BaseModel, conlist

from src.feature_extractor import extract_features


# -----------------------------
# Load model and threshold
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)


# -----------------------------
# Create FastAPI app
# -----------------------------

app = FastAPI(
    title="Phishing Detection API",
    description="Detect phishing websites using LightGBM",
    version="1.0"
)


# -----------------------------
# Enable CORS
# -----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Schemas
# -----------------------------

class PhishingFeatures(BaseModel):

    features: conlist(int, min_length=30, max_length=30)


class URLInput(BaseModel):

    url: str


# -----------------------------
# Root endpoint
# -----------------------------

@app.get("/")
def home():

    return {
        "message": "Phishing Detection API is running"
    }


# -----------------------------
# Feature-based prediction
# -----------------------------

@app.post("/predict")
def predict(data: PhishingFeatures):

    X = pd.DataFrame([data.features])

    probability = model.predict_proba(X)[0][1]

    prediction = int(probability >= threshold)

    result = "phishing" if prediction == 1 else "legitimate"

    return {
        "prediction": result,
        "probability": float(probability),
        "threshold": float(threshold)
    }


# -----------------------------
# URL-based prediction
# -----------------------------

@app.post("/predict-url")
def predict_url(data: URLInput):

    features = extract_features(data.url)

    X = pd.DataFrame([features])

    probability = model.predict_proba(X)[0][1]

    prediction = int(probability >= threshold)

    result = "phishing" if prediction == 1 else "legitimate"

    return {
        "url": data.url,
        "prediction": result,
        "probability": float(probability),
        "threshold": float(threshold)
    }
