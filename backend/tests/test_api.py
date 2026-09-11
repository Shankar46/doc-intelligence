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



def test_processing_status_is_pass_when_validation_fails(monkeypatch):
    import app.services.document_service as ds

    class FakeRepo:
        def __init__(self, db): pass
        def save_result(self, *args, **kwargs): pass

    class FakeFileValidation:
        def model_dump(self):
            return {"file_type":"image/jpeg","is_supported":True,"is_readable":True,"page_count":1,"status":"PASS","reason":None}

    class FakeOCR:
        ocr_used = True
        pages = ["Invoice\nTotal Due 100"]
        layouts = [[]]

    monkeypatch.setattr(ds, "DocumentRepository", FakeRepo)
    monkeypatch.setattr(ds, "validate_file", lambda *a, **k: FakeFileValidation())
    monkeypatch.setattr(ds, "_detect_kind", lambda *a, **k: "image")
    monkeypatch.setattr(ds, "extract_text", lambda *a, **k: FakeOCR())
    monkeypatch.setattr(ds, "extract_fields", lambda *a, **k: {"total_amount":{"value":100}})
    monkeypatch.setattr(ds, "run_validation", lambda *a, **k: {"checks":[{"name":"total","status":"FAIL","calculated_value":90,"reported_value":100,"variance":-10,"formula":"x","operands":{}}],"overall_status":"FAIL","issues":["mismatch"]})

    result = ds.process_document(object(), b"bytes", "x.jpg", "image/jpeg", "invoice")
    assert result["processing_status"] == "PASS"
    assert result["validation"]["overall_status"] == "FAIL"
