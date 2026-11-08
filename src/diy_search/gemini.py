"""Gemini APIの薄いラッパー。回数制限（429）と一時的なエラー（5xx）は待って再試行する。"""

import time
from collections.abc import Callable

import httpx
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from diy_search.config import CHAT_MODEL, EMBED_MODEL
from diy_search.usage import UsageMeter

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
_RETRY_CODES = {429, 500, 502, 503, 504}
_TIMEOUT_MS = 60_000  # 応答が止まったときは60秒で打ち切って再試行する


def make_client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(timeout=_TIMEOUT_MS))


def with_retry(call: Callable[[], T], attempts: int = 5, base_wait: float = 5.0) -> T:
    for i in range(attempts):
        try:
            return call()
        except (errors.APIError, httpx.TimeoutException) as e:
            code = getattr(e, "code", "timeout")
            if (code != "timeout" and code not in _RETRY_CODES) or i == attempts - 1:
                raise
            wait = base_wait * 2**i
            print(f"  [retry] Gemini API {code}。{wait:.0f}秒待って再試行します")
            time.sleep(wait)
    raise AssertionError("unreachable")


# gemini-embedding-2 は task_type を無視する（2026-11-08に確認：指定の有無で結果が同じ）。
# 代わりに本文の先頭に用途を書く。gemini-embedding-001 は task_type で指定する。
_PREFIX = {
    "RETRIEVAL_DOCUMENT": "title: none | text: ",
    "RETRIEVAL_QUERY": "task: search result | query: ",
}


class GeminiEmbedder:
    """task_type は "RETRIEVAL_DOCUMENT"（作例）か "RETRIEVAL_QUERY"（質問）。"""

    def __init__(self, meter: UsageMeter, model: str = EMBED_MODEL, client: genai.Client | None = None):
        self.model = model
        self.meter = meter
        self.client = client or make_client()
        self.use_prefix = model.startswith("gemini-embedding-2")

    def embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        if self.use_prefix:
            texts = [_PREFIX[task_type] + t for t in texts]
            config = None
        else:
            config = types.EmbedContentConfig(task_type=task_type)
        # 文字列のリストをそのまま渡すと1つのベクトルにまとめられるため、1件ずつ Content に包む
        # （無料枠では、1件ごとに1リクエストと数えられる：1分100件まで）
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        res = with_retry(lambda: self.client.models.embed_content(
            model=self.model, contents=contents, config=config,
        ))
        # 埋め込みの応答にはトークン数が付かないので、文字数で概算する（日本語は1文字≒1トークン）
        self.meter.total.embed_tokens += sum(len(t) for t in texts)
        return [list(e.values) for e in res.embeddings]


class GeminiChat:
    """構造化出力（Pydanticの型）でLLMを呼ぶ。temperatureは0。"""

    def __init__(self, meter: UsageMeter, model: str = CHAT_MODEL, client: genai.Client | None = None):
        self.model = model
        self.meter = meter
        self.client = client or make_client()

    def generate(self, prompt: str, schema: type[M], thinking_budget: int = 0) -> M:
        res = with_retry(lambda: self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            ),
        ))
        u = res.usage_metadata
        if u is not None:
            self.meter.total.chat_input_tokens += u.prompt_token_count or 0
            self.meter.total.chat_output_tokens += (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
        if res.parsed is None:
            return schema.model_validate_json(res.text)
        return res.parsed
