import joblib
import pandas as pd


model = joblib.load("models/best_model.pkl")

threshold = joblib.load("models/threshold.pkl")


def predict(data: pd.DataFrame):
    
    prob = model.predict_proba(data)[:, 1]
    
    prediction = (prob >= threshold).astype(int)
    
    return prediction, prob
