
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
VENDORS = pd.read_csv(BASE / "data" / "vendors.csv")

def choose_vendor(required_qty: int):
    eligible = VENDORS[VENDORS["capacity"] >= required_qty].copy()
    if eligible.empty:
        return None
    eligible["score"] = (
        eligible["fulfillment_accuracy_pct"] * 0.6
        + eligible["on_time_pct"] * 0.4
    )
    return eligible.sort_values("score", ascending=False).iloc[0].to_dict()

def vendor_scorecard():
    return VENDORS.sort_values("fulfillment_accuracy_pct", ascending=False)
