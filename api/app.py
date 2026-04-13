import os
import mlflow.spark
import mlflow
from fastapi import FastAPI, HTTPException
from pyspark.sql import SparkSession
# from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

from api.schemas import FraudInput


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
MODEL_URI = "models:/fraud_model@champion"


app = FastAPI(title="Fraud Detection API")

spark = SparkSession.builder \
    .appName("FraudPredictionAPI") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.spark.load_model(MODEL_URI)


@app.get("/")
def root():
    return {"message": "Fraud Detetion API is running"}


@app.post("/predict")
def predict(data: FraudInput):
    try:

        input_dict = data.model_dump()
        df = spark.createDataFrame([input_dict])

        pred_df = model.transform(df)

        pred_df = pred_df.withColumn(
            "probability_array",
            vector_to_array(col("probability"))
        )

        row = pred_df.select(
            "prediction", "probability_array"
        ).collect()[0]

        prediction = int(row['prediction'])
        probabilites = row["probability_array"]
        fraud_probability = float(probabilites[1])

        return {
            "prediction": prediction,
            "fraud_probability": fraud_probability,
            "non_fraud_probability": float(probabilites[0])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
