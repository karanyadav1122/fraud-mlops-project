import json
import os
import time
import requests

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import NoBrokersAvailable

RAW_TOPIC = "transactions_raw"
PREDICTION_TOPIC = "fraud_predictions"

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
API_URL = os.getenv("API_URL", "http://api:8000/predict")


def create_consumer(max_retries: int = 10, delay: int = 5) -> KafkaConsumer:
    for attempt in range(1, max_retries + 1):
        try:
            consumer = KafkaConsumer(
                RAW_TOPIC,
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
            )
            print(f"Connected consumer to Kafka on attempt {attempt}")
            return consumer
        except NoBrokersAvailable:
            print(f"Kafka not ready for consumer. Retrying... ({attempt}/{max_retries})")
            time.sleep(delay)

    raise RuntimeError("Consumer could not connect to Kafka")


def create_producer(max_retries: int = 10, delay: int = 5) -> KafkaProducer:
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            )
            print(f"Connected producer to Kafka on attempt {attempt}")
            return producer
        except NoBrokersAvailable:
            print(f"Kafka not ready for producer. Retrying... ({attempt}/{max_retries})")
            time.sleep(delay)

    raise RuntimeError("Producer could not connect to Kafka")


def call_api_with_retry(payload: dict, max_retries: int = 10, delay: int = 5) -> dict:
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"API not ready or request failed. Retrying... ({attempt}/{max_retries}) | {e}")
            time.sleep(delay)

    raise RuntimeError("Could not get prediction from API after multiple retries")


def build_features(event: dict) -> dict:
    amount = event["amount"]
    card_present = event["card_present"]

    timestamp = event["timestamp"]
    tx_hour = int(timestamp[11:13])

    is_high_amount = int(amount > 2000)
    is_card_not_present = int(not card_present)
    is_night_tx = int(tx_hour < 6 or tx_hour > 22)
    is_risky_payment = int((amount > 2000) and (not card_present))
    risk_score = float(is_high_amount + is_card_not_present + is_night_tx)

    return {
        "amount": amount,
        "is_high_amount": is_high_amount,
        "is_card_not_present": is_card_not_present,
        "tx_hour": tx_hour,
        "is_night_tx": is_night_tx,
        "is_risky_payment": is_risky_payment,
        "risk_score": risk_score,
    }


def main():
    print("Starting fraud consumer...")

    consumer = create_consumer()
    producer = create_producer()

    for message in consumer:
        event = message.value
        print(f"Received: {event}")

        try:
            payload = build_features(event)
            prediction = call_api_with_retry(payload)

            output = {
                "transaction_id": event["transaction_id"],
                "features": payload,
                "prediction": prediction,
            }

            producer.send(PREDICTION_TOPIC, value=output)
            producer.flush()

            print(f"Sent prediction: {output}")

        except Exception as e:
            print(f"Error processing message: {e}")


if __name__ == "__main__":
    main()