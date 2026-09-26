from src.predict import health_band, recommendation

def test_health_band():
    assert health_band(0.10, False) == "Normal"
    assert health_band(0.50, False) == "Warning"
    assert health_band(0.80, False) == "Critical"

def test_recommendation():
    assert "monitor" in recommendation(0.10, False).lower()
    assert "maintenance" in recommendation(0.80, True).lower()
