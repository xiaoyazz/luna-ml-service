FROM python:3.10-slim

# (Optional but helpful) install build tools for numpy/scikit-learn
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code and model files
COPY main.py .
COPY ppd_rf_classifier.pkl .
COPY ppd_rf_feature_order.json .

# Cloud Run expects the service to listen on $PORT (default 8080)
ENV PORT=8080

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
