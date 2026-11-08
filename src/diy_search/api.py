"""REST API。`uv run serve` で起動し、http://127.0.0.1:8000/docs から試せる。"""

from functools import lru_cache
from typing import Literal

import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

from diy_search.agent import AgentSearcher, ParsedQuery
from diy_search.data import load_examples
from diy_search.gemini import GeminiChat, GeminiEmbedder
from diy_search.keyword import KeywordSearcher
from diy_search.models import Example, Result
from diy_search.usage import UsageMeter
from diy_search.vector import VectorSearcher

app = FastAPI(title="DIY作例ナビ（PoC）")
Method = Literal["keyword", "vector", "agent"]


class Hit(Result):
    example: Example  # 材料（materials）を含む作例の全項目


class SearchResponse(BaseModel):
    method: Method
    query: str
    results: list[Hit]
    no_match: bool
    no_match_reason: str | None = None
    parsed: ParsedQuery | None = None  # 方式3が読み取った要望


@lru_cache(maxsize=1)
def _searchers() -> dict[str, object]:
    examples = load_examples()
    meter = UsageMeter()
    vector = VectorSearcher(examples, GeminiEmbedder(meter))
    return {
        "examples": {ex.id: ex for ex in examples},
        "keyword": KeywordSearcher(examples),
        "vector": vector,
        "agent": AgentSearcher(examples, vector, GeminiChat(meter)),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(min_length=1, max_length=300), method: Method = "agent") -> SearchResponse:
    s = _searchers()
    searcher = s[method]
    results = searcher.search(q, k=5)
    hits = [Hit(**r.model_dump(), example=s["examples"][r.example_id]) for r in results]
    is_agent = method == "agent"
    return SearchResponse(
        method=method, query=q, results=hits, no_match=not hits,
        no_match_reason=searcher.last_no_match_reason if is_agent and not hits else None,
        parsed=searcher.last_parsed if is_agent else None,
    )


def run() -> None:
    uvicorn.run("diy_search.api:app", host="127.0.0.1", port=8000)
