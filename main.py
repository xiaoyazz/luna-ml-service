from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional

from google.cloud import firestore
from google.cloud import storage

import joblib
import numpy as np
import pandas as pd
import os
import time
import threading


app = FastAPI()


# ======================================================
# Firebase / Google Cloud configuration
# ======================================================

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT",
    "newlunacare"
)

MODEL_BUCKET = os.getenv(
    "MODEL_BUCKET",
    "newlunacare.firebasestorage.app"
)

MODEL_COLLECTION = "ml_models"
MODEL_DOCUMENT = "current"

# Check Firestore for a newer model every 5 minutes.
MODEL_CHECK_INTERVAL_SECONDS = 300


# ======================================================
# Google Cloud clients
# ======================================================

firestore_client = firestore.Client(
    project=PROJECT_ID
)

storage_client = storage.Client(
    project=PROJECT_ID
)


# ======================================================
# Model state
# ======================================================

artifact = None
model = None

FEATURE_COLS = None
X_COLUMNS = None
TARGET_COL = None

current_model_version = None
last_model_check = 0

model_lock = threading.Lock()


# ======================================================
# Request schema
# ======================================================

class PredictRequest(BaseModel):
    records: List[Dict[str, Optional[float]]]


# ======================================================
# Load latest model
# ======================================================

def load_latest_model(force=False):
    global artifact
    global model
    global FEATURE_COLS
    global X_COLUMNS
    global TARGET_COL
    global current_model_version
    global last_model_check

    now = time.time()

    # Avoid checking Firestore on every prediction.
    if (
        model is not None
        and not force
        and now - last_model_check
        < MODEL_CHECK_INTERVAL_SECONDS
    ):
        return

    with model_lock:
        # Check again after obtaining the lock.
        now = time.time()

        if (
            model is not None
            and not force
            and now - last_model_check
            < MODEL_CHECK_INTERVAL_SECONDS
        ):
            return

        try:
            # ------------------------------------------
            # 1. Read ml_models/current from Firestore
            # ------------------------------------------

            doc = (
                firestore_client
                .collection(MODEL_COLLECTION)
                .document(MODEL_DOCUMENT)
                .get()
            )

            if not doc.exists:
                raise RuntimeError(
                    "ml_models/current does not exist"
                )

            model_info = doc.to_dict()

            version = model_info.get("version")
            pkl_path = model_info.get("pklPath")

            if version is None:
                raise RuntimeError(
                    "Current model document has no version"
                )

            if not pkl_path:
                raise RuntimeError(
                    "Current model document has no pklPath"
                )

            last_model_check = now

            # ------------------------------------------
            # 2. Already using this version
            # ------------------------------------------

            if (
                model is not None
                and current_model_version == version
            ):
                return

            print(
                f"New model detected: version {version}"
            )

            print(
                f"Downloading model from: {pkl_path}"
            )

            # ------------------------------------------
            # 3. Download model from Firebase Storage
            # ------------------------------------------

            bucket = storage_client.bucket(
                MODEL_BUCKET
            )

            blob = bucket.blob(
                pkl_path
            )

            local_path = (
                f"/tmp/ppd_xgb_artifact_v{version}.pkl"
            )

            blob.download_to_filename(
                local_path
            )

            # ------------------------------------------
            # 4. Load artifact
            # ------------------------------------------

            new_artifact = joblib.load(
                local_path
            )

            required_keys = [
                "model",
                "feature_cols",
                "X_columns",
                "target_col",
            ]

            missing_keys = [
                key
                for key in required_keys
                if key not in new_artifact
            ]

            if missing_keys:
                raise RuntimeError(
                    f"Model artifact missing keys: "
                    f"{missing_keys}"
                )

            # ------------------------------------------
            # 5. Switch to new model only after
            #    everything loads successfully
            # ------------------------------------------

            artifact = new_artifact

            model = artifact["model"]

            FEATURE_COLS = artifact[
                "feature_cols"
            ]

            X_COLUMNS = artifact[
                "X_columns"
            ]

            TARGET_COL = artifact[
                "target_col"
            ]

            current_model_version = version

            print(
                f"Successfully loaded model "
                f"version {version}"
            )

        except Exception as error:
            # If an older model is already loaded,
            # continue using it instead of breaking
            # predictions.
            if model is not None:
                print(
                    "Warning: could not load latest "
                    f"model: {error}"
                )

                print(
                    f"Continuing with model version "
                    f"{current_model_version}"
                )

                return

            # No model has ever been loaded.
            raise


# ======================================================
# Health check
# ======================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message":
            "LunaCare XGBoost PPD risk service running",
        "model_version":
            current_model_version
    }


# ======================================================
# Feature aggregation
# ======================================================

def summarize_user_window(df):
    features = {}

    for col in FEATURE_COLS:
        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        features[f"{col}_mean"] = (
            values.mean()
        )

        features[f"{col}_std"] = (
            values.std()
        )

        features[f"{col}_min"] = (
            values.min()
        )

        features[f"{col}_max"] = (
            values.max()
        )

        x = np.arange(
            len(values)
        )

        y = values.to_numpy(
            dtype=float
        )

        valid = ~np.isnan(y)

        if valid.sum() >= 2:
            slope = np.polyfit(
                x[valid],
                y[valid],
                1
            )[0]
        else:
            slope = np.nan

        features[
            f"{col}_slope"
        ] = slope

        features[
            f"{col}_last"
        ] = values.iloc[-1]

    return features


# ======================================================
# Prediction
# ======================================================

@app.post("/predict")
def predict(req: PredictRequest):

    if not req.records:
        raise HTTPException(
            status_code=400,
            detail="No health records provided"
        )

    # ----------------------------------------------
    # Make sure latest model is loaded
    # ----------------------------------------------

    try:
        load_latest_model()

    except Exception as error:
        print(
            f"Model loading error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to load ML model"
        )

    # ----------------------------------------------
    # Prepare patient records
    # ----------------------------------------------

    df = pd.DataFrame(
        req.records
    )

    missing = [
        col
        for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message":
                    "Missing required features",

                "missing":
                    missing
            }
        )

    # Use only features expected by this model.
    df = df[
        FEATURE_COLS
    ]

    # Convert nulls / values to numbers.
    df = df.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # ----------------------------------------------
    # Daily data -> aggregated model features
    # ----------------------------------------------

    summarized = (
        summarize_user_window(df)
    )

    X = pd.DataFrame(
        [summarized]
    )

    # Ensure exact training feature order.
    X = X.reindex(
        columns=X_COLUMNS
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    risk_score = float(
        model.predict(X)[0]
    )

    return {
        "risk_score":
            risk_score,

        "target":
            TARGET_COL,

        "days_used":
            len(df),

        "model_version":
            current_model_version
    }
