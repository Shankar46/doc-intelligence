"""Basic API flow test (spec section 11): health check + empty-file rejection."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_list_documents_returns_json_shape():
    res = client.get("/api/v1/documents")
    assert res.status_code == 200
    body = res.json()
    assert "documents" in body
    assert "count" in body


def test_process_rejects_empty_file():
    res = client.post(
        "/api/v1/documents/process",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        data={"document_type": "invoice"},
    )
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "EMPTY_FILE"


def test_dashboard_page_renders():
    res = client.get("/")
    assert res.status_code == 200
    assert "Document Intelligence Platform" in res.text


def test_document_detail_page_renders():
    res = client.get("/documents/sample_invoice.pdf/view")
    assert res.status_code == 200
    assert "Document Processing Detail" in res.text

