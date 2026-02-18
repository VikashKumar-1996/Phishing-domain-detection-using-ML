import joblib

def save_model(model, path):
    
    joblib.dump(model, path)
    
    print(f"Model saved to {path}")
    
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def save_threshold(threshold):

    path = BASE_DIR / "models" / "threshold.pkl"

    joblib.dump(threshold, path)

    print(f"Threshold saved to {path}")
