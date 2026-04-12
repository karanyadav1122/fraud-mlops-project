import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
import mlflow
import mlflow.spark

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","predictions","predictions.jsonl")
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"


def load_data(spark):
  return spark.read.json(DATA_PATH)


def train_model(df):
  feature_cols = [
    
    "amount",
    "risk_score",
    "tx_hour",
    "is_high_amount",
    "is_card_not_present",
    "is_night_tx",
    "is_risky_payment"

  ]
  
  assembler = VectorAssembler(
    inputCols = feature_cols,
    outputCol = "features"
  )
  
  rf = RandomForestClassifier(
    labelCol = "label",
    featuresCol = "features",
    numTrees = 50,
    maxDepth = 5,
    seed = 42
  )
  
  
  pipeline = Pipeline(stages = [assembler, rf])
  
  model = pipeline.fit(df)
  return model


def main():
  spark = SparkSession.builder \
          .appName("RetrainModel") \
          .config("spark.sql.parquet.compression.codec", "uncompressed") \
          .getOrCreate()
          
  spark.sparkContext.setLogLevel("WARN")
  spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
  
  mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
  mlflow.set_experiment("fraud-detection-retraining")
  
  df = load_data(spark)
  
  df = df.select(
    col("features.amount").alias("amount"),
    col("features.risk_score").alias("risk_score"),
    col("features.tx_hour").alias("tx_hour"),
    col("features.is_high_amount").alias("is_high_amount"),
    col("features.is_card_not_present").alias("is_card_not_present"),
    col("features.is_night_tx").alias("is_night_tx"),
    col("features.is_risky_payment").alias("is_risky_payment"),
    col("prediction.prediction").alias("label")
     ).dropna()
  
  model = train_model(df)
  
  with mlflow.start_run():
    mlflow.log_param("model_type","RandomForest")
    mlflow.log_param("num_trees",50)
    mlflow.log_param("max_depth",5)
    mlflow.log_param(
    "feature_cols",
    "amount,risk_score,tx_hour,is_high_amount,is_card_not_present,is_night_tx,is_risky_payment"
     )
    
    mlflow.spark.log_model(model,"fraud_rf_model",
                           registered_model_name = "fraud_model")
  
  print("Random Forest model retrained and logged to mlflow")
  
  spark.stop()
  
  
if __name__ == "__main__":
  main()  
    
            
  