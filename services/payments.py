# services/payments.py
def calc_refund(amount_dollars: float, percent: float = 15.0):
    refund_amount = round(amount_dollars * (percent / 100.0), 2)
    return {"refund_percent": percent, "refund_amount": refund_amount}
