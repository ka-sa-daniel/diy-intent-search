"""方式1：キーワード検索（BM25）。"""

import bm25s

from diy_search.models import Example, Result
from diy_search.tokenizer import tokenize


def document_text(ex: Example) -> str:
    """検索対象の本文：title・purpose・description・actions。"""
    return " ".join([ex.title, ex.purpose, ex.description, " ".join(ex.actions)])


class KeywordSearcher:
    name = "keyword"

    def __init__(self, examples: list[Example]):
        self.examples = examples
        self.retriever = bm25s.BM25()  # 既定は k1=1.5, b=0.75
        self.retriever.index([tokenize(document_text(ex)) for ex in examples], show_progress=False)

    def search(self, query: str, k: int = 5) -> list[Result]:
        tokens = tokenize(query)
        if not tokens:
            return []
        docs, scores = self.retriever.retrieve([tokens], k=min(k, len(self.examples)), show_progress=False)
        results = []
        for idx, score in zip(docs[0], scores[0]):
            if score <= 0:  # 1語も一致しない作例は返さない
                continue
            ex = self.examples[int(idx)]
            results.append(Result(example_id=ex.id, title=ex.title, score=float(score)))
        return results
