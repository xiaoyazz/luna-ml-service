from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import numpy as np

app = FastAPI()

# ---- Load model + feature order once at startup ----
model = joblib.load("ppd_rf_classifier.pkl")

with open("ppd_rf_feature_order.json") as f:
    FEATURE_ORDER = json.load(f)


class PredictRequest(BaseModel):
    # The Cloud Function will send: {"features": {featureName: value, ...}}
    features: dict


@app.get("/")
def root():
    return {"status": "ok", "message": "Luna ML RF service running"}


@app.post("/predict")
def predict(req: PredictRequest):
    feats = req.features

    # Build row in the exact order expected by the model
    row = []
    for name in FEATURE_ORDER:
        value = feats.get(name, 0)  # default 0 if missing
        row.append(float(value))

    X = np.array([row])

    # Predict class
    pred = int(model.predict(X)[0])

    # Predict probabilities if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0].tolist()
    else:
        proba = None

    return {
        "class": pred,           # 0 / 1 / 2
        "probabilities": proba,  # [p0, p1, p2] or null
        "feature_order": FEATURE_ORDER,
    }
