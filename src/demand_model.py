
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "operations_history.csv"
MODEL_PATH = BASE / "models" / "demand_model.pkl"

def train():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    df["day_num"] = df["date"].dt.dayofweek
    df = df.sort_values(["site", "meal", "date"])
    df["prev_consumed"] = df.groupby(["site","meal"])["actual_consumed"].shift(1)
    df["prev_requested"] = df.groupby(["site","meal"])["requested_quantity"].shift(1)
    df = df.dropna(subset=["prev_consumed","prev_requested"]).copy()

    X = df[["site","meal","day_num","weather","event","prev_consumed","prev_requested"]]
    y = df["actual_consumed"]

    cat = ["site","meal","weather"]
    num = ["day_num","event","prev_consumed","prev_requested"]
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", "passthrough", num)
    ])

    model = Pipeline([
        ("prep", prep),
        ("rf", RandomForestRegressor(
            n_estimators=250, random_state=42, min_samples_leaf=2, n_jobs=-1
        ))
    ])
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model

def predict(payload: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([payload])
    return float(model.predict(X)[0])

if __name__ == "__main__":
    train()
