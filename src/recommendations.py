
def generate_insights(requested, predicted, received, consumed, anomaly, vendor_name):
    insights = []
    diff_pct = (requested - predicted) / max(predicted, 1) * 100
    delivery_pct = received / max(requested, 1) * 100
    wastage_pct = (received - consumed) / max(received, 1) * 100

    if abs(diff_pct) >= 5:
        direction = "above" if requested > predicted else "below"
        insights.append(
            f"Client request is {abs(diff_pct):.1f}% {direction} the ML-predicted demand. "
            f"Review the quantity before final confirmation."
        )
    else:
        insights.append(
            "Client-requested quantity is broadly aligned with the model's expected demand."
        )

    if delivery_pct < 97:
        insights.append(
            f"Delivery accuracy is {delivery_pct:.1f}%, indicating a material fulfillment variance "
            f"that should be reviewed with {vendor_name}."
        )
    else:
        insights.append(
            f"Delivery accuracy is {delivery_pct:.1f}%, within the expected operating range."
        )

    if anomaly == -1:
        insights.append(
            "Anomaly detection flagged this transaction for unusual operational behavior. "
            "Check delivery quantity, timing and consumption records."
        )
    else:
        insights.append(
            "No major operational anomaly was detected for this transaction."
        )

    if wastage_pct > 8:
        insights.append(
            f"Wastage is {wastage_pct:.1f}%. Consider reducing future preparation quantities "
            "or reviewing menu-level acceptance."
        )
    elif wastage_pct > 5:
        insights.append(
            f"Wastage is {wastage_pct:.1f}%. Monitor this site/meal combination over the next few cycles."
        )
    else:
        insights.append(
            f"Wastage is {wastage_pct:.1f}%, indicating relatively efficient utilization."
        )
    return insights
