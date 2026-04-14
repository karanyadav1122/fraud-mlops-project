import json
import time
import os
import random
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable
from faker import Faker

fake = Faker()

KAFKA_TOPIC = "transactions_raw"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")


def create_producer(max_retries: int = 10, delay: int = 5) -> KafkaProducer:
    for attempt in range(1, max_retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode("utf-8")
            )
            print(f"Connected to Kafka on attempt {attempt}")
            return producer
        except NoBrokersAvailable:
            print(
                f"Kafka not ready yet. Retrying... ({attempt}/{max_retries})")
            time.sleep(delay)

    raise RuntimeError("Could not connect to Kafka after multiple retries")


def generate_transaction():
    amount = round(random.uniform(5, 5000), 2)
    card_present = random.choice([True, False])
    payment_method = random.choice(["credit_card", "debit_card", "upi"])
    merchant_category = random.choice(
        ["electronics", "fashion", "grocery", "travel", "gaming"]
    )

    current_time = datetime.now(timezone.utc)
    timestamp = current_time.isoformat()
    tx_hour = current_time.hour

    is_high_amount = amount > 2000
    is_card_not_present = not card_present
    is_night = tx_hour < 6 or tx_hour > 22

    fraud_score = 0

    if is_high_amount:
        fraud_score += 1
    if is_card_not_present:
        fraud_score += 1
    if is_night:
        fraud_score += 1

    if fraud_score >= 2:
        is_fraud = 1 if random.random() < 0.8 else 0
    else:
        is_fraud = 1 if random.random() < 0.1 else 0

    return {
        "transaction_id": fake.uuid4(),
        "user_id": fake.uuid4(),
        "merchant_id": fake.uuid4(),
        "device_id": fake.uuid4(),
        "amount": amount,
        "currency": "USD",
        "timestamp": timestamp,
        "merchant_category": merchant_category,
        "payment_method": payment_method,
        "card_present": card_present,
        "location": {
            "city": fake.city(),
            "state": fake.state(),
            "country": "USA"
        },
        "ip_address": fake.ipv4(),
        "is_fraud": is_fraud
    }


def main():
    print("Starting fraud transaction producer...")
    producer = create_producer()

    while True:
        transaction = generate_transaction()
        print(f"Sending: {transaction}")
        producer.send(KAFKA_TOPIC, value=transaction)
        producer.flush()
        time.sleep(2)


if __name__ == "__main__":
    main()
