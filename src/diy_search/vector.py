"""方式2：ベクトル検索。作例30件なのでDBは使わず、numpyでコサイン類似度を計算する。"""

import hashlib
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from diy_search.config import CACHE_DIR
from diy_search.models import Example, Result


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]: ...


def document_text(ex: Example) -> str:
    """埋め込む本文：title＋purpose＋description。"""
    return "\n".join([ex.title, ex.purpose, ex.description])


class EmbeddingCache:
    """キーは「モデル名・用途・本文のハッシュ」。作例を直せばハッシュが変わり、自動で計算し直す。"""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}

    @staticmethod
    def key(model: str, task_type: str, text: str) -> str:
        return f"{model}:{task_type}:{hashlib.sha256(text.encode()).hexdigest()}"

    def get_or_compute(self, embedder: Embedder, texts: list[str], task_type: str) -> np.ndarray:
        keys = [self.key(embedder.model, task_type, t) for t in texts]
        missing = [t for t, k in zip(texts, keys) if k not in self.data]
        if missing:
            for t, v in zip(missing, embedder.embed(missing, task_type)):
                self.data[self.key(embedder.model, task_type, t)] = v
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data))
        return np.array([self.data[k] for k in keys], dtype=np.float32)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


class VectorSearcher:
    name = "vector"

    def __init__(self, examples: list[Example], embedder: Embedder,
                 cache_path: Path = CACHE_DIR / "embeddings.json"):
        self.examples = examples
        self.embedder = embedder
        cache = EmbeddingCache(cache_path)
        self.doc_vectors = _normalize(cache.get_or_compute(
            embedder, [document_text(ex) for ex in examples], "RETRIEVAL_DOCUMENT"))

    def scores(self, query: str) -> np.ndarray:
        """全作例とのコサイン類似度（質問の埋め込みはキャッシュしない。応答時間を正しく測るため）。"""
        q = _normalize(np.array(self.embedder.embed([query], "RETRIEVAL_QUERY")[0], dtype=np.float32))
        return self.doc_vectors @ q

    def search(self, query: str, k: int = 5, allowed_ids: set[str] | None = None) -> list[Result]:
        """allowed_ids を渡すと、その作例だけから選ぶ（方式3の除外に使う）。"""
        sims = self.scores(query)
        order = np.argsort(-sims)
        results = []
        for i in order:
            ex = self.examples[int(i)]
            if allowed_ids is not None and ex.id not in allowed_ids:
                continue
            results.append(Result(example_id=ex.id, title=ex.title, score=float(sims[i])))
            if len(results) == k:
                break
        return results
