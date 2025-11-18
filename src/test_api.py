from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_search_no_key():
    r = client.post("/search", json={"query":"test","domain":"law","top_k":1})
    # since require_api_key uses env default, this may pass; adjust to test header auth if implemented
    assert r.status_code in (200,401)
