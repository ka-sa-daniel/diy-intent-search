"""REST API。"""

import uvicorn
from fastapi import FastAPI

app = FastAPI(title="DIY作例ナビ（PoC）")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run("diy_search.api:app", host="127.0.0.1", port=8000)
