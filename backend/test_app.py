from typing import Dict

from app import app


def _sample_payload() -> Dict[str, object]:
    return {
        "Size": "Medium",
        "Exercise Requirements (hrs/day)": 2.0,
        "Good with Children": "Yes",
        "Intelligence Rating (1-10)": 7,
        "Training Difficulty (1-10)": 3,
        "Shedding Level": "Moderate",
        "Health Issues Risk": "Low",
        "Type": "Herding",
        "Friendly Rating (1-10)": 8,
        "Life Span": 12,
        "Average Weight (kg)": 20,
    }


def test_recommend_valid_request():
    client = app.test_client()
    response = client.post("/api/recommend?top_k=3", json=_sample_payload())
    assert response.status_code == 200
    data = response.get_json()
    assert data is not None
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) > 0
    assert "user_profile" in data


def test_recommend_missing_field_returns_400():
    client = app.test_client()
    payload = _sample_payload()
    payload.pop("Size")
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data is not None
    assert "error" in data


