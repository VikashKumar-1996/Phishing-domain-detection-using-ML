import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import optuna

from src.data_loader import load_data
from src.config import *
from src.utils import save_model
from src.threshold import find_optimal_threshold
from src.explain import generate_shap_summary


# MODEL DEFINITIONS


def build_models():
    
    models = {
        
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000))
        ]),
        
        "random_forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        
        "decision_tree": Pipeline([
            ("model", DecisionTreeClassifier(
                max_depth=10,
                random_state=RANDOM_STATE
            ))
        ]),
        
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                kernel="rbf"
            ))
        ]),
        
        "ann": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=RANDOM_STATE
            ))
        ]),
        
        "lightgbm": Pipeline([
            ("model", lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=7,
                random_state=RANDOM_STATE
            ))
        ])
    }
    
    return models

# CROSS VALIDATION


def evaluate_models(models, X_train, y_train):
    
    results = {}
    
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    
    print("\nCross-validation results:")
    
    for name, model in models.items():
        
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )
        
        mean_score = scores.mean()
        
        results[name] = mean_score
        
        print(f"{name:15} ROC-AUC: {mean_score:.4f}")
    
    return results
# LIGHTGBM HYPERPARAMETER TUNING


def tune_lightgbm(X_train, y_train):
    
    def objective(trial):
        
        params = {
            
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            
            "random_state": RANDOM_STATE
        }
        
        model = lgb.LGBMClassifier(**params)
        
        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        )
        
        score = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        ).mean()
        
        return score
    
    
    study = optuna.create_study(direction="maximize")
    
    study.optimize(objective, n_trials=30)
    
    print("\nBest LightGBM params:")
    print(study.best_params)
    
    best_model = lgb.LGBMClassifier(**study.best_params)
    
    return best_model



# MAIN TRAINING PIPELINE


def main():
    
    print("Loading data...")
    
    df = load_data()
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    
    print("Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    
    
    models = build_models()
    
    
    results = evaluate_models(models, X_train, y_train)
    
    
    # tune LightGBM
    print("\nTuning LightGBM...")
    
    tuned_lgbm = tune_lightgbm(X_train, y_train)
    
    tuned_lgbm.fit(X_train, y_train)
    
    
    y_prob = tuned_lgbm.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_prob)

    print(f"Test ROC-AUC: {test_auc:.4f}")


# find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_prob)
    from src.utils import save_threshold

    save_threshold(optimal_threshold)



# generate SHAP summary
    generate_shap_summary(
        tuned_lgbm,
        X_train,
        save_path="models/shap_summary.png"
    )

    if __name__ == "__main__":
        
        main()
