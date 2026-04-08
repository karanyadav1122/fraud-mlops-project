import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField ,StringType, DoubleType, \
  BooleanType, IntegerType
  
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR, "data", "bronze")
SILVER_PATH = os.path.join(BASE_DIR, "data", "silver")
SILVER_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "silver")
QUARANTINE_PATH = os.path.join(BASE_DIR, "data", "quarantine")
QUARANTINE_CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "quarantine")



spark = SparkSession.builder \
        .appName("FraudSilverStream") \
        .getOrCreate()
        
        
spark.sparkContext.setLogLevel("WARN")

bronze_schema = StructType([
    StructField("json_str", StringType(), True)
])

transaction_schema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("merchant_id", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("currency", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("payment_method", StringType(), True),
    StructField("card_present", BooleanType(), True),
    StructField("location", StructType([
        StructField("city", StringType(), True),
        StructField("state", StringType(), True),
        StructField("country", StringType(), True)
    ]), True),
    StructField("ip_address", StringType(), True),
    StructField("is_fraud", IntegerType(), True)
])


df_bronze = spark.readStream \
            .schema(bronze_schema) \
            .json(BRONZE_PATH)
            
df_parsed = df_bronze.select(
    col("json_str"),
    from_json(col("json_str"), transaction_schema).alias("data")
)


df_valid = df_parsed.select("data.*").filter(
  col("data").isNotNull() &
    col("data.transaction_id").isNotNull() &
    col("data.user_id").isNotNull() &
    col("data.merchant_id").isNotNull() &
    col("data.device_id").isNotNull() &
    col("data.amount").isNotNull() &
    (col("data.amount") > 0) &
    col("data.currency").isNotNull() &
    col("data.timestamp").isNotNull()
)

df_invalid = df_parsed.filter(
  col("data").isNull() |
    col("data.transaction_id").isNull() |
    col("data.user_id").isNull() |
    col("data.merchant_id").isNull() |
    col("data.device_id").isNull() |
    col("data.amount").isNull() |
    (col("data.amount") <= 0) |
    col("data.currency").isNull() |
    col("data.timestamp").isNull()
).select("json_str")

valid_query = df_valid.writeStream \
              .format("json") \
              .outputMode("append") \
              .option("path",SILVER_PATH) \
              .option("checkpointLocation", SILVER_CHECKPOINT) \
              .start()
              
invalid_query = df_invalid.writeStream \
                .format("json") \
                .outputMode("append") \
                .option("path",QUARANTINE_PATH) \
                .option("checkpointLocation",QUARANTINE_CHECKPOINT) \
                .start()
                
                
spark.streams.awaitAnyTermination()

                        
                  
                              
              
              
                
                    