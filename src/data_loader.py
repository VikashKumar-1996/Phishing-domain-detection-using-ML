import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_data():
    
    dataset = fetch_ucirepo(id=327)
    
    X = dataset.data.features
    y = dataset.data.targets
    
    df = pd.concat([X, y], axis=1)
    
    # convert target
    df["result"] = df["result"].map({-1: 1, 1: 0})
    
    return df
