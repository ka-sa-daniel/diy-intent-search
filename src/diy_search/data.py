"""作例・質問・正解の読み込み。"""

import csv
import json
from pathlib import Path

from diy_search.config import DATA_DIR, EVAL_DIR
from diy_search.models import Example, Query


def load_examples(path: Path = DATA_DIR / "examples.json") -> list[Example]:
    return [Example.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def load_queries(name: str) -> list[Query]:
    """name は "dev" か "test"。"""
    path = EVAL_DIR / f"queries_{name}.json"
    return [Query.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def load_judgments(path: Path = EVAL_DIR / "judgments.csv") -> dict[tuple[str, str], int]:
    """(query_id, example_id) -> grade。書いていない組は0とみなすので、ここには入れない。"""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f, skipinitialspace=True)
        return {(r["query_id"].strip(), r["example_id"].strip()): int(r["grade"]) for r in rows}
