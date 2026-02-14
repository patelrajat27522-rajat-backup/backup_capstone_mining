import numpy as np 


def classify_risk(pred):
    if pred < 2.0:
        return "Low"
    elif pred < 3.0:
        return "Medium"
    else:
        return "High"
    
def get_top_drivers(shap_values_row, feature_names, top_k=4):
    values = shap_values_row.values
    idx = np.argsort(np.abs(values))[::-1][:top_k]

    drivers = []
    for i in idx:
        drivers.append({
            "feature": feature_names[i],
            "impact": float(values[i]),
        })
    return drivers


def enrich_driver_info(drivers, row_data):
    enriched = []
    for d in drivers:
        direction = "increasing silica" if d["impact"] > 0 else "reducing silica"
        enriched.append({
            "feature": d["feature"],
            "value": float(row_data[d["feature"]]),
            "impact": round(d["impact"], 3),
            "direction": direction
        })
    return enriched