from typing import Dict, Any

FEATURE_SCHEMA: Dict[str, type] = {
    "transaction_id": str,
    "amount": float,
    "is_high_amount": int,
    "is_card_not_present": int,
    "tx_hour": int,
    "is_night_tx": int,
    "is_risky_payment": int,
    "risk_score": float,
    "is_fraud": int,
    "timestamp": str,
}


def validate_features(record: Dict[str, Any]) -> bool:
    for key, expected_type in FEATURE_SCHEMA.items():
        if key not in record:
            raise ValueError(f"Missing feature: {key}")

        if not isinstance(record[key], expected_type):
            raise TypeError(
                f"Feature '{key}' expected {expected_type}, got {type(record[key])}"
            )

    return True
