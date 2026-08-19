
from pathlib import Path
import sys
import math
import pandas as pd
import streamlit as st
import plotly.express as px

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from src.demand_model import predict as demand_predict
from src.anomaly_detection import score as anomaly_score
from src.vendor_engine import choose_vendor, vendor_scorecard
from src.recommendations import generate_insights

st.set_page_config(
    page_title="F&B Ops Intelligence",
    page_icon="🍱",
    layout="wide"
)

@st.cache_data
def load_data():
    df = pd.read_csv(BASE / "data" / "operations_history.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("🍱 F&B Ops Intelligence")
st.caption("AI-powered demand planning, vendor ordering, delivery reconciliation and operational anomaly detection")

tabs = st.tabs([
    "📊 Dashboard",
    "🤖 Demand Planner",
    "📦 Order Generator",
    "🚚 Reconciliation",
    "🏭 Vendor Intelligence"
])

# ---------------- Dashboard ----------------
with tabs[0]:
    latest = df[df["date"] >= df["date"].max() - pd.Timedelta(days=6)].copy()
    total_req = int(latest["requested_quantity"].sum())
    total_rec = int(latest["actual_received"].sum())
    total_con = int(latest["actual_consumed"].sum())
    wastage = total_rec - total_con
    delivery_acc = total_rec / max(total_req, 1) * 100

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Meals Requested", f"{total_req:,}")
    c2.metric("Meals Received", f"{total_rec:,}")
    c3.metric("Meals Consumed", f"{total_con:,}")
    c4.metric("Wastage", f"{wastage:,}")
    c5.metric("Delivery Accuracy", f"{delivery_acc:.1f}%")

    st.subheader("Operational Trend")
    daily = latest.groupby("date", as_index=False).agg(
        requested=("requested_quantity","sum"),
        received=("actual_received","sum"),
        consumed=("actual_consumed","sum")
    )
    fig = px.line(daily, x="date", y=["requested","received","consumed"],
                  markers=True, title="Requested vs Received vs Consumed")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Exceptions")
    recent = latest.copy()
    recent["delivery_variance_pct"] = (
        (recent["requested_quantity"] - recent["actual_received"])
        / recent["requested_quantity"].replace(0,1) * 100
    )
    exceptions = recent[recent["delivery_variance_pct"] >= 5].sort_values(
        "delivery_variance_pct", ascending=False
    ).head(10)
    if exceptions.empty:
        st.success("No high-severity delivery exceptions in the recent period.")
    else:
        st.dataframe(
            exceptions[[
                "date","site","meal","vendor","requested_quantity",
                "actual_received","delivery_variance_pct"
            ]].rename(columns={"delivery_variance_pct":"Delivery Variance %"}),
            use_container_width=True
        )

# ---------------- Demand Planner ----------------
with tabs[1]:
    st.subheader("AI Demand Planner")
    st.write("Enter a client requirement and context. The ML model estimates expected consumption.")
    col1, col2, col3 = st.columns(3)
    with col1:
        site = st.selectbox("Site", sorted(df["site"].unique()))
        meal = st.selectbox("Meal", sorted(df["meal"].unique()))
        requested = st.number_input("Client requested quantity", min_value=40, max_value=5000, value=850, step=10)
    with col2:
        event = st.selectbox("Event / special activity", ["No", "Yes"])
        weather = st.selectbox("Weather", sorted(df["weather"].unique()))
        day = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    with col3:
        history_site = df[(df["site"]==site) & (df["meal"]==meal)].sort_values("date")
        prev_consumed = int(history_site["actual_consumed"].iloc[-1])
        prev_requested = int(history_site["requested_quantity"].iloc[-1])
        st.metric("Previous consumed", f"{prev_consumed:,}")
        st.metric("Previous requested", f"{prev_requested:,}")

    day_num = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day)
    payload = {
        "site": site,
        "meal": meal,
        "day_num": day_num,
        "weather": weather,
        "event": 1 if event=="Yes" else 0,
        "prev_consumed": prev_consumed,
        "prev_requested": prev_requested
    }
    if st.button("🔮 Predict Demand", type="primary"):
        pred = demand_predict(payload)
        recommended = int(round(pred / 10) * 10)
        difference = requested - pred
        difference_pct = difference / max(pred,1) * 100
        a,b,c = st.columns(3)
        a.metric("ML Predicted Demand", f"{pred:,.0f}")
        b.metric("Recommended Order", f"{recommended:,}")
        c.metric("Client Request vs Prediction", f"{difference_pct:+.1f}%")
        if difference_pct > 5:
            st.warning("Client request is materially above model-predicted demand.")
        elif difference_pct < -5:
            st.info("Client request is below the model-predicted demand.")
        else:
            st.success("Client request is broadly aligned with model-predicted demand.")

# ---------------- Order Generator ----------------
with tabs[2]:
    st.subheader("Smart Vendor Order Generator")
    st.write("Create a recommended order using the AI quantity and vendor performance.")
    order_site = st.selectbox("Order site", sorted(df["site"].unique()), key="order_site")
    order_meal = st.selectbox("Order meal", sorted(df["meal"].unique()), key="order_meal")
    order_qty = st.number_input("AI / approved quantity", min_value=40, max_value=5000, value=850, step=10)
    chosen = choose_vendor(int(order_qty))
    if chosen:
        st.write("Recommended vendor")
        a,b,c = st.columns(3)
        a.metric("Vendor", chosen["vendor"])
        b.metric("Fulfillment Accuracy", f'{chosen["fulfillment_accuracy_pct"]:.1f}%')
        c.metric("On-time Performance", f'{chosen["on_time_pct"]:.1f}%')
        order = pd.DataFrame([{
            "Site": order_site,
            "Meal": order_meal,
            "Recommended Quantity": int(order_qty),
            "Vendor": chosen["vendor"],
            "Vendor Capacity": int(chosen["capacity"])
        }])
        st.dataframe(order, use_container_width=True)
        st.success("Order generated successfully. This represents the digital replacement for manual vendor communication.")
    else:
        st.error("No single vendor has sufficient capacity. Split-order logic is required.")

# ---------------- Reconciliation ----------------
with tabs[3]:
    st.subheader("Delivery & Consumption Reconciliation")
    a,b,c,d = st.columns(4)
    with a: ordered = st.number_input("Ordered", 0, 5000, 880, 10)
    with b: dispatched = st.number_input("Dispatched", 0, 5000, 880, 10)
    with c: received = st.number_input("Received", 0, 5000, 822, 10)
    with d: consumed = st.number_input("Consumed", 0, 5000, 790, 10)
    vendor_name = st.selectbox("Vendor", sorted(df["vendor"].unique()), key="recon_vendor")

    if st.button("🔍 Reconcile & Analyze", type="primary"):
        delivery_var = max(0, ordered - received)
        delivery_var_pct = delivery_var / max(ordered,1) * 100
        wastage_qty = max(0, received - consumed)
        wastage_pct = wastage_qty / max(received,1) * 100

        feature_record = {
            "delivery_variance_pct": delivery_var_pct,
            "delivery_delay_min": max(0, dispatched-received),
            "wastage_pct": wastage_pct
        }
        label, score = anomaly_score(feature_record)

        x1,x2,x3,x4 = st.columns(4)
        x1.metric("Delivery Variance", f"{delivery_var} meals")
        x2.metric("Delivery Accuracy", f"{received/max(ordered,1)*100:.1f}%")
        x3.metric("Wastage", f"{wastage_qty} meals")
        x4.metric("Wastage %", f"{wastage_pct:.1f}%")

        if label == -1:
            st.error("🔴 Anomaly detected: this transaction is outside normal operating patterns.")
        else:
            st.success("🟢 No major anomaly detected.")

        st.subheader("AI Operations Insights")
        insights = generate_insights(ordered, max(ordered,1), received, consumed, label, vendor_name)
        for i in insights:
            st.write("• " + i)

# ---------------- Vendor Intelligence ----------------
with tabs[4]:
    st.subheader("Vendor Intelligence")
    scorecard = vendor_scorecard().copy()
    scorecard["overall_score"] = (
        scorecard["fulfillment_accuracy_pct"] * 0.6
        + scorecard["on_time_pct"] * 0.4
    )
    st.dataframe(scorecard.round(1), use_container_width=True)

    fig = px.bar(
        scorecard,
        x="vendor",
        y="overall_score",
        title="Vendor Overall Performance Score",
        text="overall_score"
    )
    st.plotly_chart(fig, use_container_width=True)
    best = scorecard.iloc[0]
    st.info(
        f"Recommendation: {best['vendor']} currently ranks highest on the combined "
        f"fulfillment + on-time performance score."
    )

st.divider()
st.caption("Prototype created as an AI-driven decision-support system. Operational data shown is synthetic and intended for demonstration.")
