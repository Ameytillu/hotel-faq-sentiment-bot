# tests/test_policy.py
from core.policy.restaurant_actions import decide_action

def test_positive_coupon():
    a, _ = decide_action("positive", 0.95)
    assert a == "COUPON_FREE"

def test_negative_refund():
    a, _ = decide_action("negative", 0.10)
    assert a == "REFUND_15"

def test_neutral_none():
    a, _ = decide_action("neutral", 0.55)
    assert a == "NONE"
