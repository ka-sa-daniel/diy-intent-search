"""設定。モデル名は環境変数で切り替えられる。APIキーはここで読まない（SDKが GEMINI_API_KEY を読む）。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

CHAT_MODEL = os.environ.get("DIY_CHAT_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.environ.get("DIY_EMBED_MODEL", "gemini-embedding-2")

# 有料で使った場合の料金（USD / 100万トークン）。評価の「1問あたりのAPI費用」の換算に使う。
# 公式の料金表で確認してから入れる。未設定（None）の間は費用を「未設定」と表示する。
PRICE_CHAT_INPUT: float | None = None
PRICE_CHAT_OUTPUT: float | None = None
PRICE_EMBED: float | None = None
