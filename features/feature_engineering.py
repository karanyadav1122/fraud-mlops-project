def build_features_from_event(event: dict) -> dict:
  amount = float(event['amount'])
  card_present = bool(event["card_present"])
  timestamp = event["timestamp"]
  tx_hour = int(timestamp[11:13])
  
  is_high_amount = int(amount > 2000)
  is_card_not_present = int(not card_present)
  is_night_tx = int(tx_hour < 6 or tx_hour > 22)
  is_risky_payment = int(is_high_amount and is_card_not_present)
  risk_score = float(is_high_amount + is_card_not_present + is_night_tx)
  
  
  return {
        "transaction_id": event["transaction_id"],
        "amount": amount,
        "is_high_amount": is_high_amount,
        "is_card_not_present": is_card_not_present,
        "tx_hour": tx_hour,
        "is_night_tx": is_night_tx,
        "is_risky_payment": is_risky_payment,
        "risk_score": risk_score,
        "is_fraud": event.get("is_fraud"),
        "timestamp": timestamp,
    }
  
  
  
  