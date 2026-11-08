from fastapi.testclient import TestClient

from diy_search.api import app


def test_health() -> None:
    res = TestClient(app).get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_search_keyword_without_api_key(monkeypatch) -> None:
    from diy_search import api
    from diy_search.data import load_examples
    from diy_search.keyword import KeywordSearcher

    examples = load_examples()
    monkeypatch.setattr(api, "_searchers", lambda: {
        "examples": {ex.id: ex for ex in examples}, "keyword": KeywordSearcher(examples)})
    res = TestClient(app).get("/search", params={"q": "網戸を張り替える", "method": "keyword"})
    assert res.status_code == 200
    body = res.json()
    assert body["results"] and body["results"][0]["example"]["materials"]
    assert body["parsed"] is None
