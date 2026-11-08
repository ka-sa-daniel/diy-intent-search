"""データの型。"""

from typing import Literal, Protocol

from pydantic import BaseModel


class Size(BaseModel):
    width: float | None = None
    depth: float | None = None
    height: float | None = None


class Material(BaseModel):
    name: str
    qty: str
    note: str = ""


class Example(BaseModel):
    id: str
    title: str
    category: str
    purpose: str
    actions: list[str]
    location: str
    size_cm: Size
    max_load_kg: float | None
    constraints: list[str]
    tools: list[str]
    description: str
    materials: list[Material]


QueryType = Literal[
    "purpose_only", "purpose_size", "problem_only", "keyword",
    "constraint", "vague", "no_match", "own",
]


class Query(BaseModel):
    id: str
    type: QueryType
    text: str


class Result(BaseModel):
    example_id: str
    title: str
    score: float
    reason: str | None = None  # 方式3だけが書く
    size_note: str | None = None  # 方式3だけが書く（寸法の違い）


class Searcher(Protocol):
    """3方式の共通インターフェース。方式3が「該当なし」と判断したときは空のリストを返す。"""

    name: str

    def search(self, query: str, k: int = 5) -> list[Result]: ...
