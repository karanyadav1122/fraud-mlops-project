import os

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(BASE_DIR, "data", "gold")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_rf_pipeline")

spark = (
    SparkSession.builder
    .appName("FraudModelTraining")
    .config("spark.sql.parquet.compression.codec", "uncompressed")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.parquet.compression.codec", "uncompressed")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-detection-training")

print("MLflow tracking URI:", mlflow.get_tracking_uri())

feature_cols = [
    "amount",
    "is_high_amount",
    "is_card_not_present",
    "tx_hour",
    "is_night_tx",
    "is_risky_payment",
    "risk_score",
]

df = spark.read.json(GOLD_PATH)

df_model = df.select(*feature_cols, "is_fraud").dropna()
df_model = df_model.withColumn("label", col("is_fraud").cast("double"))

train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=50,
    maxDepth=5,
    seed=42,
)

pipeline = Pipeline(stages=[assembler, rf])

auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC",
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1",
)

acc_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
)

with mlflow.start_run():
    print("MLflow run started")

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("num_trees", 50)
    mlflow.log_param("max_depth", 5)

    model = pipeline.fit(train_df)
    pred_df = model.transform(test_df)

    auc = auc_eval.evaluate(pred_df)
    f1 = f1_eval.evaluate(pred_df)
    acc = acc_eval.evaluate(pred_df)

    mlflow.log_metric("auc", auc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("accuracy", acc)

    model_info = mlflow.spark.log_model(
        spark_model=model,
        artifact_path="fraud_rf_model",
        registered_model_name="fraud_model",
    )

    print("\n=== Fraud model metrics ===")
    print(f"AUC: {auc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Model URI: {model_info.model_uri}")

    pred_df.select(
        "label",
        "prediction",
        "probability",
        "amount",
        "risk_score",
    ).show(20, truncate=False)

    model.write().overwrite().save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

spark.stop()
