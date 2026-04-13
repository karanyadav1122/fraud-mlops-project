import os
import json
from statistics import mean


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_FILE = os.path.join(
    BASE_DIR, "data", "predictions", "predictions.jsonl")


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
    tx_hours = [r["features"]["tx_hour"] for r in records]
    risk_scores = [r["features"]["risk_score"] for r in records]
    fraud_probs = [r["prediction"]["fraud_probability"] for r in records]
    predictions = [r["prediction"]["prediction"] for r in records]

    return {
        "count": len(records),
        "avg_amount": safe_mean(amounts),
        "avg_tx_hour": safe_mean(tx_hours),
        "avg_risk_score": safe_mean(risk_scores),
        "avg_fraud_probability": safe_mean(fraud_probs),
        "fraud_prediction_rate": safe_mean(predictions),
    }


def compare(baseline, recent):
    report = {}
    for key in baseline:
        if key == "count":
            continue

        base_val = baseline[key]
        recent_val = recent.get(key, 0.0)
        diff = recent_val - base_val
        pct_change = (diff / base_val * 100) if base_val != 0 else 0.0

        DRIFT_THRESHOLD = 10

        report[key] = {
            "baseline": round(base_val, 4),
            "recent": round(recent_val, 4),
            "diff": round(diff, 4),
            "pct_change": round(pct_change, 2),
            "drift_detected": abs(pct_change) > DRIFT_THRESHOLD
        }
        if abs(pct_change) > DRIFT_THRESHOLD:
            print(f" Drift detected in {key}: {pct_change:.2f}%")

    return report


def main():

    records = load_predictions(PREDICTIONS_FILE)

    if len(records) < 20:
        print("Not enough predictions record yet for drift analysis")

        return

    split_idx = int(len(records)*.7)
    baseline_records = records[:split_idx]
    recent_records = records[split_idx:]

    baseline_summary = summarize(baseline_records)
    recent_summary = summarize(recent_records)

    drift_report = compare(baseline_summary, recent_summary)

    drift_flag = any([
        drift_report["avg_risk_score"]["drift_detected"],
        drift_report["avg_amount"]["drift_detected"]
    ])

    if drift_flag:
        print("Drift detected..retraining model")
        os.system("python training/retrain_model.py")

    else:
        print("no significant drift. Retraining not triggered")

    print("\n=== Baseline Summary ===")
    print(json.dumps(baseline_summary, indent=2))

    print("\n=== Recent Summary ===")
    print(json.dumps(recent_summary, indent=2))

    print("\n=== Drift Report ===")
    print(json.dumps(drift_report, indent=2))


if __name__ == "__main__":
    main()
