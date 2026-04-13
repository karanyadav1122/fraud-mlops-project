import json
from kafka import KafkaConsumer, KafkaProducer

RAW_TOPIC = "transactions_raw"
VALID_TOPIC = "transactions_valid"
DLQ_TOPIC = "transactions_dead_letter"

BOOTSTRAP_SERVERS = "localhost:9092"

consumer = KafkaConsumer(
    RAW_TOPIC,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True
)

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode("utf-8")
)


def is_valid_transaction(tx):

    try:

        if tx['amount'] <= 0:
            return False

        if tx['currency'] != "USD":
            return False

        if not tx["transaction_id"]:
            return False

        return True

    except Exception:
        return False


def main():
    print("starting consumer..")

    for message in consumer:
        tx = message.value

        print(f"received: {tx}")

        if is_valid_transaction(tx):
            print("valid transaction, forwarding to vlalid topic")
            producer.send(VALID_TOPIC, value=tx)

        else:
            print("invalid transaction, sending to dead letter queue")
            producer.send(DLQ_TOPIC, value=tx)


if __name__ == "__main__":
    main()
