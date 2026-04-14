import os
import mlflow.spark
import mlflow
from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

from api.schemas import FraudInput

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MODEL_URI = "models:/fraud_model/1"

app = FastAPI(title="Fraud Detection API")

spark = SparkSession.builder \
    .appName("FraudPredictionAPI") \
    .config("spark.sql.parquet.compression.codec", "uncompressed") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

model = None


def get_model():
    global model

    if model is not None:
        return model

    try:
        model = mlflow.spark.load_model(MODEL_URI)
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}


@app.post("/predict")
def predict(data: FraudInput):
    try:
        loaded_model = get_model()

        if loaded_model is None:
            raise HTTPException(status_code=500, detail="Model not available")

        input_dict = data.model_dump()
        df = spark.createDataFrame([input_dict])

        pred_df = loaded_model.transform(df)

        pred_df = pred_df.withColumn(
            "probability_array",
            vector_to_array(col("probability"))
        )

        row = pred_df.select(
            "prediction", "probability_array"
        ).collect()[0]

        prediction = int(row["prediction"])
        probabilities = row["probability_array"]
        fraud_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "fraud_probability": fraud_probability,
            "non_fraud_probability": float(probabilities[0])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
