import joblib
import pandas as pd
from pathlib import Path


# get project root
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"


# load model safely
model = joblib.load(MODEL_PATH)

threshold = joblib.load(THRESHOLD_PATH)


def predict(data: pd.DataFrame):
    
    prob = model.predict_proba(data)[:, 1]
    
    prediction = (prob >= threshold).astype(int)
    
    return prediction, prob


# test run
if __name__ == "__main__":
    
    print("Model loaded successfully")
