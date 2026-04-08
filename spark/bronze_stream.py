import os
from pyspark.sql import SparkSession

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRONZE_PATH = os.path.join(BASE_DIR,"data","bronze")
CHECKPOINT_PATH = os.path.join(BASE_DIR,"checkpoints","bronze")


spark = SparkSession.builder \
        .appName("FraudBronzeStream") \
          .getOrCreate()
          

spark.sparkContext.setLogLevel("WARN")

df_kafka = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers","localhost:9092") \
    .option("subscribe","transactions_raw") \
    .option("startingOffsets","latest") \
    .load()
    
df_bronze = df_kafka.selectExpr("CAST(value AS STRING) as json_str")

query = df_bronze.writeStream \
        .format("json") \
        .outputMode("append") \
        .option("path",BRONZE_PATH) \
        .option("checkpointLocation",CHECKPOINT_PATH) \
        .start()       

query.awaitTermination()

                      