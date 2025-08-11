# core/policy/restaurant_actions.py
POS_THRESHOLD = 0.70
NEG_THRESHOLD = 0.30

def decide_action(label, score):
    # Coerce anything (int/np.int32/etc.) to string safely
    lbl = str(label).lower()

    if lbl == "negative" and score <= NEG_THRESHOLD:
        return "REFUND_15", f"🙁 Negative ({score:.2f}). We’re sorry—offering a 15% refund."
    if lbl == "positive" and score >= POS_THRESHOLD:
        return "COUPON_FREE", f"😊 Positive ({score:.2f}). Thanks! Enjoy a free coupon."
    return "NONE", f"😐 {lbl.capitalize()} ({score:.2f})."
