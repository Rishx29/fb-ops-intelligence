
from src.demand_model import train as train_demand
from src.anomaly_detection import train as train_anomaly

if __name__ == "__main__":
    train_demand()
    train_anomaly()
    print("Models trained successfully.")
