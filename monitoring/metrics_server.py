import json
import time
import os
from statistics import mean
from prometheus_client import Gauge, start_http_server

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_FILE = os.path.join(
    BASE_DIR, "data", "predictions", "Predictions.jsonl")

DRIFT_THRESHOLD = 10

avg_amount_pct_change = Gauge(
    "fraud_avg_amount_pct_change", "Percent change in average amount")
avg_risk_score_pct_change = Gauge(
    "fraud_avg_risk_score_pct_change", "Percent change in average risk score")
avg_fraud_probability_pct_change = Gauge(
    "fraud_avg_fraud_probability_pct_change", "Percent change in average fraud probability")
fraud_prediction_rate_pct_change = Gauge(
    "fraud_prediction_rate_pct_change", "percent change in fraud prediction rate")

avg_amount_drift_detected = Gauge(
    "fraud_avg_amount_drift_detected", "1 if avg amount drift detected else 0")
avg_risk_score_drift_detected = Gauge(
    "fraud_avg_risk_score_drift_detected", "1 if avg risk score drift detected else 0")
avg_fraud_probability_drift_detected = Gauge(
    "fraud_avg_fraud_probability_drift_detected", "1 if avg fraud probability drift detected else 0")
fraud_prediction_rate_drift_detected = Gauge(
    "fraud_prediction_rate_drift_detected", "1 if fraud prediction rate drift detected else 0")


def load_predictions(filepath):
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def safe_mean(values):
    return mean(values) if values else 0.0


def summarize(records):
    if not records:
        return {}

    amounts = [r["features"]["amount"] for r in records]
    risk_scores = [r["features"]["risk_score"] for r in records]
    fraud_probs = [r["prediction"]["fraud_probability"] for r in records]
    predictions = [r["prediction"]["prediction"] for r in records]

    return {
        "avg_amount": safe_mean(amounts),
        "avg_risk_score": safe_mean(risk_scores),
        "avg_fraud_probability": safe_mean(fraud_probs),
        "fraud_prediction_rate": safe_mean(predictions),
    }


def pct_change(baseline, recent):
    if baseline == 0:
        return 0.0
    return ((recent - baseline) / baseline) * 100


def update_metrics():
    records = load_predictions(PREDICTIONS_FILE)

    if len(records) < 20:
        return

    split_idx = int(len(records)*.7)
    baseline_records = records[:split_idx]
    recent_records = records[split_idx:]

    baseline = summarize(baseline_records)
    recent = summarize(recent_records)

    amount_change = pct_change(baseline["avg_amount"], recent["avg_amount"])
    risk_change = pct_change(
        baseline["avg_risk_score"], recent["avg_risk_score"])
    fraud_prob_change = pct_change(
        baseline["avg_fraud_probability"], recent["avg_fraud_probability"])
    pred_rate_change = pct_change(
        baseline["fraud_prediction_rate"], recent["fraud_prediction_rate"])

    avg_amount_pct_change.set(amount_change)
    avg_risk_score_pct_change.set(risk_change)
    avg_fraud_probability_pct_change.set(fraud_prob_change)
    fraud_prediction_rate_pct_change.set(pred_rate_change)

    avg_amount_drift_detected.set(
        1 if abs(amount_change) > DRIFT_THRESHOLD else 0)
    avg_risk_score_drift_detected.set(
        1 if abs(risk_change) > DRIFT_THRESHOLD else 0)
    avg_fraud_probability_drift_detected.set(
        1 if abs(fraud_prob_change) > DRIFT_THRESHOLD else 0)
    fraud_prediction_rate_drift_detected.set(
        1 if abs(pred_rate_change) > DRIFT_THRESHOLD else 0)


if __name__ == "__main__":
    start_http_server(9000)
    print("Metrics server running on http://localhost:9000/metrics")

    while True:
        update_metrics()
        time.sleep(15)
