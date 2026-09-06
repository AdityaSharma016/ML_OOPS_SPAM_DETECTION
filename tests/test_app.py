from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"text": "Congratulations! You won a free prize. Click now!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "result" in data
    assert "confidence" in data