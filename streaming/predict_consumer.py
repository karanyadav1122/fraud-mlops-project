import json
import os
import requests
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timezone

KAFKA_INPUT_TOPIC = "transactions_raw"
KAFKA_OUTPUT_TOPIC = "fraud_predictions"
KAFKA_BOOTSTRAP_SERVER = "localhost:9092"


API_URL = "http://localhost:8000/predict"
MODEL_RUN_ID = os.getenv("MODEL_RUN_ID","unknown")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(BASE_DIR,"data","predictions")
PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "predictions.jsonl")

os.makedirs(PREDICTIONS_DIR,exist_ok= True)



consumer = KafkaConsumer(
          KAFKA_INPUT_TOPIC,
          bootstrap_servers = KAFKA_BOOTSTRAP_SERVER,
          value_deserializer = lambda x : json.loads(x.decode("utf-8")),
          auto_offset_reset = "latest",
          enable_auto_commit = True,
)

producer = KafkaProducer(
            bootstrap_servers = KAFKA_BOOTSTRAP_SERVER,
            value_serializer = lambda x : json.dumps(x).encode('utf-8'),
)

def build_features(tx):
    amount = tx["amount"]
    card_present = tx["card_present"]
    payment_method = tx["payment_method"]
    timestamp = tx["timestamp"]

    
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    tx_hour = dt.hour

    is_high_amount = int(amount > 2000)
    is_card_not_present = int(not card_present)
    is_night_tx = int(tx_hour < 6 or tx_hour > 22)
    is_risky_payment = int(payment_method == "credit_card")

    risk_score = (
        is_high_amount
        + is_card_not_present
        + is_night_tx
        + is_risky_payment
    )

    return {
        "amount": amount,
        "is_high_amount": is_high_amount,
        "is_card_not_present": is_card_not_present,
        "tx_hour": tx_hour,
        "is_night_tx": is_night_tx,
        "is_risky_payment": is_risky_payment,
        "risk_score": float(risk_score),
    }
    
def save_prediction_record(record):
  with open(PREDICTIONS_FILE, "a",encoding="utf-8") as f:
    f.write(json.dumps(record)+ "\n")
    
    
print("Starting prediction consumer ")    

for message in consumer:
  tx = message.value
  print(f"Received: {tx}")
  
  features = build_features(tx)
  
  try:
    response = requests.post(API_URL, json= features, timeout= 10)
    if response.status_code == 200:
      prediction = response.json()
      
      prediction_record = {
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
                "model_run_id": MODEL_RUN_ID,
                "transaction_id": tx.get("transaction_id"),
                "transaction": tx,
                "features": features,
                "prediction": prediction,
      }    
      
      producer.send(KAFKA_OUTPUT_TOPIC, value= prediction_record)
      producer.flush()
      
      save_prediction_record(prediction_record)
      
      print(f"Save Prediction record: {prediction_record}")

    else:
            print(f"API error: {response.text}")

  except Exception as e:
        print(f"Error calling API: {e}")