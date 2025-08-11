# services/coupons.py
import uuid
from datetime import datetime, timedelta

def create_free_coupon(days_valid: int = 30):
    code = "MEAL-" + uuid.uuid4().hex[:8].upper()
    expires = (datetime.utcnow() + timedelta(days=days_valid)).date().isoformat()
    return {"code": code, "expires": expires, "percent_off": 100}
