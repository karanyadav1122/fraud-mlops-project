import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour, when
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType, BooleanType


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(BASE_DIR, 'data', 'gold')
SILVER_PATH = os.path.join(BASE_DIR, "data", "silver")
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'gold')

spark = SparkSession.builder \
    .appName("GoldStream") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

silver_schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("user_id", StringType()) \
    .add("merchant_id", StringType()) \
    .add("device_id", StringType()) \
    .add("amount", DoubleType()) \
    .add("currency", StringType()) \
    .add("timestamp", StringType()) \
    .add("merchant_category", StringType()) \
    .add("payment_method", StringType()) \
    .add("card_present", BooleanType()) \
    .add("ip_address", StringType()) \
    .add("is_fraud", IntegerType())

df_silver = spark.readStream \
    .schema(silver_schema) \
    .json(SILVER_PATH)

df = df_silver.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"))
)

df_features = df \
    .withColumn("tx_hour", hour(col("timestamp")))  \
    .withColumn("is_night_tx", when(col("tx_hour") < 6, 1).otherwise(0)) \
    .withColumn("is_high_amount", when(col("amount") > 3000, 1).otherwise(0)) \
    .withColumn("is_card_not_present", when(~col("card_present"), 1).otherwise(0))  \
    .withColumn("is_risky_payment", when(col("payment_method") == "credit_card", 1).otherwise(0))

df_features = df_features.withColumn(
    "risk_score",
    col("is_high_amount") +
    col("is_card_not_present") +
    col("is_risky_payment") +
    col("is_night_tx")
)


query = df_features.writeStream \
    .format("json") \
    .outputMode("append")  \
    .option("path", GOLD_PATH) \
    .option("checkpointLocation", CHECKPOINT_PATH) \
    .start()

query.awaitTermination()
