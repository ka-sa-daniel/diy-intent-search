"""Gemini APIの薄いラッパー。回数制限（429）と一時的なエラー（5xx）は待って再試行する。"""

import time
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai import errors, types

from diy_search.config import EMBED_MODEL
from diy_search.usage import UsageMeter

T = TypeVar("T")
_RETRY_CODES = {429, 500, 503}


def with_retry(call: Callable[[], T], attempts: int = 5, base_wait: float = 5.0) -> T:
    for i in range(attempts):
        try:
            return call()
        except errors.APIError as e:
            if e.code not in _RETRY_CODES or i == attempts - 1:
                raise
            wait = base_wait * 2**i
            print(f"  [retry] Gemini API {e.code}。{wait:.0f}秒待って再試行します")
            time.sleep(wait)
    raise AssertionError("unreachable")


class GeminiEmbedder:
    """task_type は "RETRIEVAL_DOCUMENT"（作例）か "RETRIEVAL_QUERY"（質問）。"""

    def __init__(self, meter: UsageMeter, model: str = EMBED_MODEL, client: genai.Client | None = None):
        self.model = model
        self.meter = meter
        self.client = client or genai.Client()

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        # 文字列のリストをそのまま渡すと1つのベクトルにまとめられるため、1件ずつ Content に包む
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        res = with_retry(lambda: self.client.models.embed_content(
            model=self.model, contents=contents, config=types.EmbedContentConfig(task_type=task_type),
        ))
        # 埋め込みの応答にはトークン数が付かないので、文字数で概算する（日本語は1文字≒1トークン）
        self.meter.total.embed_tokens += sum(len(t) for t in texts)
        return [list(e.values) for e in res.embeddings]
