# LunaCare ML Service

Machine learning service for my capstone project **LunaCare**.

This service provides Postpartum Depression (PPD) risk predictions for the LunaCare Doctor Portal. It is implemented as a Python API and deployed using Google Cloud Run.

The project currently includes two generations of machine learning models:

1. **Random Forest Classifier** — original version
2. **XGBoost Regressor** — current version

---

## ML Model Versions

### Version 1 — Random Forest Classifier

The original LunaCare ML service used a Random Forest classifier.

The related files are:

```text
ppd_rf_classifier.pkl
ppd_rf_classifier_v2.pkl
ppd_rf_feature_order.json
ppd_rf_feature_order_v2.json
```

The newer Random Forest version used 22 input features:

```text
age
mood_1to5
bleeding_1to10
hair_loss_1to10
appetite_issue_1to10
sleep_trouble_1to10
sleep_hours
deep_sleep_hours
rem_sleep_hours
sleep_efficiency_pct
wake_after_sleep_onset_min
avg_heart_rate_bpm
resting_heart_rate_bpm
hrv_sdnn_ms
vo2max_ml_kg_min
steps
active_energy_kcal
exercise_minutes
stand_hours
oxygen_saturation_pct
respiratory_rate_bpm
weight_kg
```

The API accepted one set of patient features.

Example request:

```json
{
  "features": {
    "age": 30,
    "mood_1to5": 2,
    "bleeding_1to10": 3,
    "hair_loss_1to10": 4,
    "appetite_issue_1to10": 5,
    "sleep_trouble_1to10": 7,
    "sleep_hours": 5.8,
    "deep_sleep_hours": 1.0,
    "rem_sleep_hours": 1.2,
    "sleep_efficiency_pct": 78,
    "wake_after_sleep_onset_min": 45,
    "avg_heart_rate_bpm": 76,
    "resting_heart_rate_bpm": 72,
    "hrv_sdnn_ms": 42,
    "vo2max_ml_kg_min": 31,
    "steps": 3200,
    "active_energy_kcal": 300,
    "exercise_minutes": 20,
    "stand_hours": 9,
    "oxygen_saturation_pct": 98,
    "respiratory_rate_bpm": 16,
    "weight_kg": 65
  }
}
```

The Random Forest model returned one of three classes:

| Class | Meaning |
|---|---|
| 0 | Stable |
| 1 | Medium Risk |
| 2 | High Risk |

Example response:

```json
{
  "class": 2,
  "probabilities": [0.05, 0.20, 0.75]
}
```

This was the original ML service used by LunaCare.

---

## Version 2 — XGBoost PPD Risk Model

The current LunaCare ML service uses an XGBoost regression model.

The model is stored in:

```text
ppd_xgb_artifact.pkl
```

Unlike the Random Forest version, the XGBoost model does not predict one of three classes. Instead, it predicts a continuous PPD risk score.

The prediction target is:

```text
ppd_risk_score_0to1
```

Example output:

```json
{
  "risk_score": 0.5763
}
```

The XGBoost model is designed to analyze changes in health and behavioural data across multiple days.

> The risk score is a model-generated indicator for the LunaCare capstone project and should not be treated as a medical diagnosis.

---

## XGBoost Input Data

The current XGBoost model uses 28 daily measurements.

### Self-Reported Data

```text
fatigue_1to10
mood_1to5
bleeding_1to10
hair_loss_1to10
appetite_issue_1to10
sleep_trouble_1to10
```

### Sleep Data

```text
sleep_hours
deep_sleep_hours
rem_sleep_hours
core_sleep_hours
sleep_efficiency_pct
wake_after_sleep_onset_min
```

### Cardiovascular and Fitness Data

```text
avg_heart_rate_bpm
resting_heart_rate_bpm
walking_heart_rate_avg_bpm
hrv_sdnn_ms
respiratory_rate_bpm
oxygen_saturation_pct
vo2max_ml_kg_min
```

### Activity Data

```text
steps
distance_walked_km
flights_climbed
active_energy_kcal
basal_energy_kcal
exercise_minutes
stand_hours
sunlight_hours
```

### Body Data

```text
weight_kg
```

---

## Feature Engineering

The XGBoost model analyzes a window of daily health records.

For each of the 28 daily measurements, the service calculates six summary features:

```text
mean
standard deviation
minimum
maximum
slope
last value
```

This creates:

```text
28 daily measurements × 6 summary statistics = 168 model features
```

For example, `sleep_hours` is transformed into:

```text
sleep_hours_mean
sleep_hours_std
sleep_hours_min
sleep_hours_max
sleep_hours_slope
sleep_hours_last
```

The same process is applied to all 28 daily measurements. The resulting 168 features are passed into the XGBoost model.

---

## Model Comparison

| Feature | Random Forest | XGBoost |
|---|---|---|
| Model type | Classifier | Regressor |
| Status | Legacy | Current |
| Input style | One set of current values | Multiple days of health data |
| Raw input features | 22 | 28 daily features |
| Feature engineering | None in API | Mean, std, min, max, slope, last |
| Final model features | 22 | 168 |
| Output | Class 0 / 1 / 2 | Continuous risk score |
| Example output | `2` / High Risk | `0.5763` |

---

## Current ML Flow

```text
LunaCare iOS App
        │
        │ HealthKit and questionnaire data
        ▼
     Firestore
        │
        │ Patient health history
        ▼
Firebase Functions Backend
        │
        │ Multiple days of patient data
        ▼
Google Cloud Run
luna-ml-service
        │
        ▼
Feature Engineering
        │
        │ 28 daily measurements
        ▼
168 summarized features
        │
        ▼
XGBoost Model
        │
        ▼
PPD Risk Score
        │
        ▼
LunaCare Doctor Portal
```

---

## API

The ML service is implemented using FastAPI.

When running locally, FastAPI documentation is available at:

```text
http://localhost:8080/docs
```

### Health Check

#### Request

```http
GET /
```

#### Example Response

```json
{
  "status": "ok",
  "message": "LunaCare XGBoost PPD risk service running"
}
```

---

## PPD Risk Prediction

### Request

```http
POST /predict
```

The endpoint accepts multiple days of patient health data. The request contains a `records` array.

Example:

```json
{
  "records": [
    {
      "fatigue_1to10": 7,
      "mood_1to5": 3,
      "bleeding_1to10": 2,
      "hair_loss_1to10": 4,
      "appetite_issue_1to10": 5,
      "sleep_trouble_1to10": 7,
      "sleep_hours": 6.2,
      "deep_sleep_hours": 1.1,
      "rem_sleep_hours": 1.3,
      "core_sleep_hours": 3.8,
      "sleep_efficiency_pct": 82,
      "wake_after_sleep_onset_min": 35,
      "avg_heart_rate_bpm": 75,
      "resting_heart_rate_bpm": 70,
      "walking_heart_rate_avg_bpm": 95,
      "hrv_sdnn_ms": 45,
      "respiratory_rate_bpm": 16,
      "oxygen_saturation_pct": 98,
      "vo2max_ml_kg_min": 32,
      "steps": 4200,
      "distance_walked_km": 3.1,
      "flights_climbed": 4,
      "active_energy_kcal": 350,
      "basal_energy_kcal": 1450,
      "exercise_minutes": 22,
      "stand_hours": 10,
      "sunlight_hours": 1.2,
      "weight_kg": 65
    }
  ]
}
```

The service converts the daily records into the 168 model features before making a prediction.

### Example Prediction Response

```json
{
  "risk_score": 0.5763,
  "target": "ppd_risk_score_0to1",
  "days_used": 30
}
```

- `risk_score` is the continuous output produced by the XGBoost regression model.
- `days_used` represents the number of daily records included in the prediction.

---

## Project Structure

```text
luna-ml-service/
│
├── Dockerfile
├── main.py
├── requirements.txt
├── ppd_xgb_artifact.pkl
├── ppd_rf_classifier.pkl
├── ppd_rf_classifier_v2.pkl
├── ppd_rf_feature_order.json
└── ppd_rf_feature_order_v2.json
```

---

## File Descriptions

### `main.py`

Main FastAPI application.

Responsibilities include:

- Loading the current XGBoost model
- Receiving prediction requests
- Validating input data
- Creating summarized model features
- Running the XGBoost prediction
- Returning the PPD risk score

### `ppd_xgb_artifact.pkl`

Current XGBoost model artifact.

The artifact contains:

```text
model
feature_cols
target_col
X_columns
```

These values are used to ensure the API prepares prediction inputs in the same format expected by the trained model.

### `Dockerfile`

Defines the Docker container used to run the ML API.

The container installs the required Python dependencies and starts the FastAPI service using Uvicorn.

### `requirements.txt`

Contains the Python packages required by the service.

Current dependencies include:

```text
fastapi
uvicorn
numpy
pandas
joblib
xgboost
scikit-learn
```

---

## Legacy Random Forest Files

The following files belong to the previous Random Forest implementation:

```text
ppd_rf_classifier.pkl
ppd_rf_classifier_v2.pkl
ppd_rf_feature_order.json
ppd_rf_feature_order_v2.json
```

These files are retained for project history and reference.

The current API uses:

```text
ppd_xgb_artifact.pkl
```

---

## Run Locally

### Requirements

Docker must be installed and running.

### Build the Docker Image

From the `luna-ml-service` directory:

```bash
docker build -t luna-ml-service .
```

### Run the Docker Container

```bash
docker run -p 8080:8080 luna-ml-service
```

The service will be available at:

```text
http://localhost:8080
```

### Verify the Service

Open:

```text
http://localhost:8080
```

Expected response:

```json
{
  "status": "ok",
  "message": "LunaCare XGBoost PPD risk service running"
}
```

### FastAPI Swagger Documentation

Open:

```text
http://localhost:8080/docs
```

The Swagger interface can be used to test:

```text
GET /
POST /predict
```

---

## Google Cloud Deployment

The LunaCare ML service is deployed using Google Cloud Run.

Google Cloud project:

```text
fir-example-bd842
```

Cloud Run service:

```text
luna-ml-service
```

Region:

```text
us-central1
```

### Select the Google Cloud Project

```bash
gcloud config set project fir-example-bd842
```

### Deploy to Cloud Run

From inside the `luna-ml-service` directory:

```bash
gcloud run deploy luna-ml-service \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

Deploying to the same Cloud Run service creates a new revision while keeping the existing service name.

---

## Backend Integration

The ML service is used by the LunaCare Firebase Functions backend.

The backend retrieves patient data from Firestore and sends the required patient health records to the Cloud Run `/predict` endpoint.

Expected architecture:

```text
Doctor Portal
      │
      ▼
Firebase Functions
      │
      ▼
Firestore
      │
      │ Patient health records
      ▼
Firebase Functions
      │
      │ Multiple daily records
      ▼
Cloud Run
luna-ml-service
      │
      ▼
Feature Engineering
      │
      ▼
XGBoost Model
      │
      ▼
PPD Risk Score
      │
      ▼
Doctor Portal
```

---

## Technologies

- Python
- FastAPI
- XGBoost
- scikit-learn
- pandas
- NumPy
- joblib
- Docker
- Google Cloud Run
- Firebase
- Firestore

---

## LunaCare

LunaCare is my capstone project focused on postpartum health monitoring.

The system combines:

- Apple Watch health data
- Apple HealthKit data
- Self-reported mood tracking
- Postpartum symptom tracking
- Machine learning
- Firebase and Firestore
- A Doctor Portal for healthcare providers

The machine learning component is designed to provide additional risk insights based on changes in physiological and behavioural health data over time.

> **Note:** The ML-generated PPD risk score is intended as a decision-support feature for the LunaCare capstone project and is not a medical diagnosis.
