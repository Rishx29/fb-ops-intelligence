
# F&B Ops Intelligence

## AI-Powered F&B Demand, Ordering & Reconciliation System

### Overview

F&B Ops Intelligence is a prototype decision-support platform designed around a real-world food-service workflow:

**Client requirement → AI demand recommendation → vendor order → delivery → reconciliation → anomaly detection → operational insight**

The project is inspired by operational processes commonly found in institutional/corporate food-service environments. All data used in the prototype is synthetic.

### Business problem

F&B operations often rely on manual transfer of meal requirements, vendor communication, delivery checks and end-of-day reconciliation. This can create:

- over/under-ordering
- delivery discrepancies
- excess food and wastage
- weak visibility into vendor performance
- slow operational decision-making

### Solution

The prototype digitizes this workflow and adds machine-learning decision support.

### AI components

1. **Demand prediction**
   - Random Forest Regressor
   - Predicts expected meal consumption from site, meal, day, weather, event and recent demand.

2. **Anomaly detection**
   - Isolation Forest
   - Detects unusual combinations of delivery variance, delivery delay proxy and wastage.

3. **AI-style operational recommendations**
   - Converts model and reconciliation outputs into human-readable operational actions.

### Core modules

- Dashboard
- AI Demand Planner
- Smart Vendor Order Generator
- Delivery & Consumption Reconciliation
- Vendor Intelligence

### Tech stack

- Python
- Pandas / NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

### Run locally

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

### Project structure

```text
fb-ops-intelligence/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/
│   ├── operations_history.csv
│   └── vendors.csv
├── models/
│   ├── demand_model.pkl
│   └── anomaly_model.pkl
└── src/
    ├── demand_model.py
    ├── anomaly_detection.py
    ├── vendor_engine.py
    ├── recommendations.py
    └── __init__.py
```

### Example user journey

1. Client enters: Site A, Lunch, 850 meals, Friday, event = Yes.
2. Demand model predicts approximately 878 meals.
3. System recommends an order quantity near the forecast.
4. Vendor engine identifies the highest-scoring vendor with sufficient capacity.
5. Operator records ordered, dispatched, received and consumed quantities.
6. Reconciliation calculates variance and wastage.
7. Isolation Forest flags unusual operational behaviour.
8. Recommendation engine produces actionable insights.

### Future enhancements

- OCR-based extraction from vendor delivery challans
- API/WhatsApp/email-based order notifications
- Multi-vendor order splitting
- LLM-powered root-cause analysis
- Inventory integration
- Role-based access control
- Cloud database and authentication
- Time-series models such as XGBoost/Prophet/LSTM for advanced forecasting

### Important note

This is a demonstration prototype using synthetic data. It is not connected to any real company's systems or confidential records.
