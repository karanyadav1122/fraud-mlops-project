#  Fraud Detection MLOps Pipeline

[![CI](https://github.com/karanyadav1122/fraud-mlops-project/actions/workflows/ci.yml/badge.svg)](https://github.com/karanyadav1122/fraud-mlops-project/actions/workflows/ci.yml)

---

## Overview

This project implements an **end-to-end real-time fraud detection system** using modern MLOps practices.

It simulates financial transactions, processes them through a streaming data pipeline, serves real-time predictions via an API, and tracks experiments using MLflow.

---

##  Architecture
Kafka Producer
↓
Kafka Topic (transactions_raw)
↓
Spark Bronze (raw ingestion)
↓
Spark Silver (validation + quarantine)
↓
Spark Gold (feature engineering)
↓
Model Training (PySpark ML)
↓
MLflow (tracking + registry)
↓
FastAPI (real-time inference)
↓
Predictions + Logging
↓
Monitoring + Drift Detection
↓
Auto Retraining (future)

---

## Project Structure

fraud-mlops-project/
│
├── api/
│ ├── app.py
│ └── schemas.py
│
├── streaming/
│ ├── producer.py
│ └── consumer.py
│
├── spark/
│ ├── bronze_stream.py
│ ├── silver_stream.py
│ └── gold_stream.py
│
├── training/
│ ├── train_model.py
│ └── retrain_model.py
│
├── monitoring/
│ ├── drift_report.py
│ └── metrics_server.py
│
├── tests/
│
├── Dockerfile
├── .dockerignore
├── .github/workflows/ci.yml
│
├── data/ # ignored
├── checkpoints/ # ignored
├── mlruns/ # ignored
├── models/ # ignored
│
└── README.md

---

## Technologies Used

- Apache Kafka (Streaming)
- PySpark Structured Streaming
- FastAPI (API serving)
- MLflow (Experiment Tracking & Model Registry)
- Python
- Docker
- GitHub Actions (CI/CD)
- Faker (Data Simulation)

---

##  Data Pipeline

###  Producer
- Generates synthetic transaction data
- Sends data to Kafka topic: `transactions_raw`

###  Bronze Layer
- Reads raw Kafka data
- Stores JSON as-is
- Acts as raw ingestion layer

###  Silver Layer
- Parses JSON into structured schema
- Filters invalid records
- Stores:
  - valid → `data/silver`
  - invalid → `data/quarantine`

###  Gold Layer
- Feature engineering:
  - `is_high_amount`
  - `is_card_not_present`
  - `is_night_tx`
  - `risk_score`
- Outputs ready-to-train dataset

---

##  Real-Time Inference (FastAPI)

- Exposes `/predict` endpoint
- Accepts transaction payload
- Applies trained Spark model
- Returns fraud probability

### Example Request:

```json
{
  "amount": 3500,
  "is_high_amount": 1,
  "is_card_not_present": 1,
  "tx_hour": 23,
  "is_night_tx": 1,
  "is_risky_payment": 1,
  "risk_score": 4.5
}

Example Response:

{
  "prediction": 1,
  "fraud_probability": 0.82,
  "non_fraud_probability": 0.18
}

Model Training
Model: RandomForestClassifier
Framework: PySpark ML
Features:
Transaction amount
Behavioral flags
Engineered risk score
Metrics:
AUC
F1-score
Accuracy
MLflow Integration

Tracks:

Parameters
Metrics
Model artifacts

Run MLflow UI

mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

Open:

http://localhost:5000

Monitoring & Drift Detection
Tracks:

Average transaction amount
Risk score distribution
Fraud prediction rate

Detects drift using:
Percentage change thresholds


Example:

if pct_change > threshold:
    drift_detected = True

CI/CD Pipeline

GitHub Actions workflow includes:
Linting (flake8)
Unit testing (pytest)
Docker build validation

Runs automatically on:
push
pull_request    

How to Run (End-to-End)

1. Start Kafka

docker-compose up -d kafka zookeeper

2. Run Streaming Pipeline

spark-submit spark/bronze_stream.py
spark-submit spark/silver_stream.py
spark-submit spark/gold_stream.py

3. Train Model
spark-submit training/train_model.py

4. Start API
uvicorn api.app:app --host 0.0.0.0 --port 8000

5. Send Prediction Request
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{...}'

Author

Karan Yadav