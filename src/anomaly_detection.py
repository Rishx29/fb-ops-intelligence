
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "operations_history.csv"
MODEL_PATH = BASE / "models" / "anomaly_model.pkl"

FEATURES = [
    "delivery_variance_pct",
    "delivery_delay_min",
    "wastage_pct"
]

def train():
    df = pd.read_csv(DATA)
    df["delivery_variance_pct"] = (
        (df["requested_quantity"] - df["actual_received"])
        / df["requested_quantity"].replace(0, 1) * 100
    )
    df["wastage_pct"] = (
        (df["actual_received"] - df["actual_consumed"])
        / df["actual_received"].replace(0, 1) * 100
    )
    X = df[FEATURES].fillna(0)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("iso", IsolationForest(
            n_estimators=250, contamination=0.04, random_state=42
        ))
    ])
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    return model

def score(record: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([record])[FEATURES].fillna(0)
    return int(model.predict(X)[0]), float(model.decision_function(X)[0])

if __name__ == "__main__":
    train()
