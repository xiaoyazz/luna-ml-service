from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd


app = FastAPI()


# Load XGBoost artifact once when service starts
artifact = joblib.load("ppd_xgb_artifact.pkl")

model = artifact["model"]
FEATURE_COLS = artifact["feature_cols"]
X_COLUMNS = artifact["X_columns"]
TARGET_COL = artifact["target_col"]


class PredictRequest(BaseModel):
    records: List[Dict[str, float]]


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "LunaCare XGBoost PPD risk service running"
    }


def summarize_user_window(df):
    features = {}

    for col in FEATURE_COLS:
        values = pd.to_numeric(df[col], errors="coerce")

        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()

        x = np.arange(len(values))
        y = values.to_numpy(dtype=float)

        valid = ~np.isnan(y)

        if valid.sum() >= 2:
            slope = np.polyfit(x[valid], y[valid], 1)[0]
        else:
            slope = np.nan

        features[f"{col}_slope"] = slope
        features[f"{col}_last"] = values.iloc[-1]

    return features


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="No health records provided"
        )

    df = pd.DataFrame(req.records)

    missing = [
        col
        for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing": missing
            }
        )

    # Use only the columns required by the ML model
    df = df[FEATURE_COLS]

    # Convert values to numbers
    df = df.apply(pd.to_numeric, errors="coerce")

    # Convert daily records into the 168 aggregated features
    summarized = summarize_user_window(df)

    X = pd.DataFrame([summarized])

    # Make sure feature order matches training
    X = X.reindex(columns=X_COLUMNS)

    risk_score = float(model.predict(X)[0])

    return {
        "risk_score": risk_score,
        "target": TARGET_COL,
        "days_used": len(df)
    }
