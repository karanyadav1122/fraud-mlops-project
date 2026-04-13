from pydantic import BaseModel


class FraudInput(BaseModel):
    amount: float
    is_high_amount: int
    is_card_not_present: int
    tx_hour: int
    is_night_tx: int
    is_risky_payment: int
    risk_score: float
