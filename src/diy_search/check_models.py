"""Gemini APIで使えるモデルと、temperature 0・構造化出力・埋め込みが動くかを確かめる。"""

from google import genai
from google.genai import types
from pydantic import BaseModel

from diy_search.config import CHAT_MODEL, EMBED_MODEL


class Probe(BaseModel):
    purpose: str
    location: str | None


def main() -> None:
    client = genai.Client()

    print("== 使えるモデル ==")
    for m in client.models.list():
        actions = ",".join(m.supported_actions or [])
        if "generateContent" in actions or "embedContent" in actions:
            print(f"{m.name}\t{actions}")

    print(f"\n== チャット（{CHAT_MODEL}、temperature 0、構造化出力） ==")
    res = client.models.generate_content(
        model=CHAT_MODEL,
        contents="玄関で傘をまとめて立てておきたい。目的と場所を抜き出して。",
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=Probe,
        ),
    )
    print(res.parsed)
    print("usage:", res.usage_metadata)

    print(f"\n== 埋め込み（{EMBED_MODEL}） ==")
    emb = client.models.embed_content(model=EMBED_MODEL, contents=["壁に棚を付けたい"])
    print("次元数:", len(emb.embeddings[0].values))
