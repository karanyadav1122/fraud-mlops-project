from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_predict_schema(monkeypatch):
    from pyspark.ml.linalg import Vectors

    class MockModel:
        def transform(self, df):
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()

            return spark.createDataFrame([{
                "prediction": 1,
                "probability": Vectors.dense([0.2, 0.8])
            }])

    monkeypatch.setattr("api.app.get_model", lambda: MockModel())

    payload = {
        "amount": 3500,
        "is_high_amount": 1,
        "is_card_not_present": 1,
        "tx_hour": 23,
        "is_night_tx": 1,
        "is_risky_payment": 1,
        "risk_score": 4.5
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200