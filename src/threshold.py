import numpy as np
from sklearn.metrics import f1_score


def find_optimal_threshold(y_true, y_prob):
    
    thresholds = np.arange(0.0, 1.0, 0.01)
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        
        y_pred = (y_prob >= threshold).astype(int)
        
        score = f1_score(y_true, y_pred)
        
        if score > best_score:
            
            best_score = score
            best_threshold = threshold
    
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_score:.4f}")
    
    return best_threshold
