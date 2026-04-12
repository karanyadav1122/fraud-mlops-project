import os
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR,"data","predictions","predictions.jsonl")
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
REGISTERED_MODEL_NAME = "fraud_model"
EXPERIMENT_NAME = "fraud-detection-retraining"


def load_data(spark):
  
  df = spark.read.json(DATA_PATH)

  df = df.select(
        col("features.amount").alias("amount"),
        col("features.risk_score").alias("risk_score"),
        col("features.tx_hour").alias("tx_hour"),
        col("features.is_high_amount").alias("is_high_amount"),
        col("features.is_card_not_present").alias("is_card_not_present"),
        col("features.is_night_tx").alias("is_night_tx"),
        col("features.is_risky_payment").alias("is_risky_payment"),
        col("prediction.prediction").cast("double").alias("label")
    ).dropna()

  return df


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
  return model, feature_cols

def evaluate_model(model, df):
  pred_df = model.transform(df)
  
  auc_eval = BinaryClassificationEvaluator(
    labelCol = "label",
    rawPredictionCol = "rawPrediction",
    metricName = "areaUnderROC"
  )
  
  f1_eval = MulticlassClassificationEvaluator(
    labelCol = "label",
    predictionCol = "prediction",
    metricName = "f1"
  )
  
  auc = auc_eval.evaluate(pred_df)
  f1 = f1_eval.evaluate(pred_df)
  return auc, f1

def get_current_champion_f1(client: MlflowClient) -> float | None:
  try:
    champion = client.get_model_version_by_alias(REGISTERED_MODEL_NAME,"champion")
    run = client.get_run(champion.run_id)
    return run.data.metrics.get("f1")
  except Exception:
    return None
  
def get_latest_registered_version(client: MlflowClient, run_id :str) -> str | None:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
      return None
    
    matching.sort(key = lambda v: int(v.version), reverse = True)
    return matching[0].version


def main():
  spark = SparkSession.builder \
          .appName("RetrainModel") \
          .config("spark.sql.parquet.compression.codec", "uncompressed") \
          .getOrCreate()
          
  spark.sparkContext.setLogLevel("WARN")
  spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")
  
  mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
  mlflow.set_experiment(EXPERIMENT_NAME)
  
  client = MlflowClient(tracking_uri= MLFLOW_TRACKING_URI)
  
  
  df = load_data(spark)
  model, feature_cols = train_model(df)
  auc, f1 = evaluate_model(model, df)
  
  current_champion_f1 = get_current_champion_f1(client)
  
  
    
  
  with mlflow.start_run() as run:
    mlflow.log_param("model_type","RandomForest")
    mlflow.log_param("num_trees", 50)
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("feature_cols",",".join(feature_cols))
    
    mlflow.log_metric("auc",auc)
    mlflow.log_metric("f1",f1)
    
    mlflow.spark.log_model(
      model,
      "fraud_rf_model",
      registered_model_name = REGISTERED_MODEL_NAME
    ) 
    
    run_id = run.info.run_id
    
  new_version = get_latest_registered_version(client, run_id)
  
  print(f"Candidate metrics -> AUC: {auc:.4f}, F1: {f1:.4f}")
  print(f"Current champion F1 -> {current_champion_f1}")
  
  if new_version is None:
    print("could not find registered version for the new run.")
   
  elif current_champion_f1 is None or f1 > current_champion_f1:
    client.set_registered_model_alias(
      name= REGISTERED_MODEL_NAME,
      alias= "champion",
      version = new_version
    )
    
    print(f"promoted version {new_version} to @champion")
    
  else:
    print(f"Kept existing champion. New vesion {new_version} was not better")
    
    
  spark.stop()
  
if __name__ == "__main__":
  main()      
    