# 🚀 Fraud Detection MLOps Pipeline

## 📌 Overview

This project implements an **end-to-end real-time fraud detection system** using modern MLOps practices.

It simulates financial transactions, processes them through a streaming data pipeline, and trains a machine learning model with experiment tracking using MLflow.

---

## 🏗️ Architecture
Kafka Producer
↓
Kafka Topic (transactions_raw)
↓
Spark Bronze Layer (raw ingestion)
↓
Spark Silver Layer (clean + validation + quarantine)
↓
Spark Gold Layer (feature engineering)
↓
Model Training (PySpark ML)
↓
MLflow (experiment tracking + model logging)


---

## 📂 Project Structure


fraud-mlops-project/
│
├── streaming/
│ ├── producer.py
│ └── consumer.py (optional)
│
├── spark/
│ ├── bronze_stream.py
│ ├── silver_stream.py
│ └── gold_stream.py
│
├── training/
│ └── train_model.py
│
├── data/ # ignored (generated data)
├── checkpoints/ # ignored (Spark checkpoints)
├── mlruns/ # ignored (MLflow artifacts)
├── models/ # ignored (local model storage)
│
├── .gitignore
└── README.md


---

## ⚙️ Technologies Used

- Apache Kafka (Streaming)
- PySpark Structured Streaming
- Python
- MLflow (Experiment Tracking)
- RandomForest (PySpark ML)
- Faker (Data Simulation)

---

## 🔄 Data Pipeline

### 1️⃣ Producer
- Generates synthetic transaction data
- Sends data to Kafka topic: `transactions_raw`

### 2️⃣ Bronze Layer
- Reads raw Kafka data
- Stores JSON as-is
- Acts as raw ingestion layer

### 3️⃣ Silver Layer
- Parses JSON into structured schema
- Filters invalid records
- Stores:
  - valid → `data/silver`
  - invalid → `data/quarantine`

### 4️⃣ Gold Layer
- Feature engineering:
  - `is_high_amount`
  - `is_card_not_present`
  - `is_night_tx`
  - `risk_score`
- Outputs ready-to-train dataset

---

## 🤖 Model Training

- Model: **RandomForestClassifier**
- Features:
  - amount
  - behavioral flags
  - engineered risk score

### 📊 Metrics:
- AUC
- F1-score
- Accuracy

---

## 📊 MLflow Integration

Tracks:
- Parameters
- Metrics
- Model artifacts

### ▶️ Run MLflow UI

```bash
mlflow ui --backend-store-uri file:///path/to/mlruns --host 0.0.0.0 --port 5000

Open in browser:

http://localhost:5000
▶️ How to Run
1. Start Kafka

(Start Zookeeper and Kafka services)

2. Run Producer
python streaming/producer.py
3. Run Bronze Stream
spark-submit \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.13:4.1.1 \
  spark/bronze_stream.py
4. Run Silver Stream
spark-submit spark/silver_stream.py
5. Run Gold Stream
spark-submit spark/gold_stream.py
6. Train Model
spark-submit training/train_model.py

Key Features

Real-time streaming pipeline
Medallion architecture (Bronze/Silver/Gold)
Data validation + quarantine handling
Feature engineering in streaming
ML training with PySpark
Experiment tracking with MLflow


Future Improvements

FastAPI real-time inference service
Kafka consumer for predictions
Feature store integration (Feast)
Model registry (MLflow)
Drift detection & monitoring
Docker + CI/CD + Kubernetes deployment


Key Learnings
Streaming data pipelines with Kafka + Spark
Managing offsets and checkpoints
Medallion architecture design
ML experiment tracking with MLflow
Feature engineering in real-time systems

 Author

Karan Yadav