"""設定。モデル名は環境変数で切り替えられる。APIキーはここで読まない（SDKが GEMINI_API_KEY を読む）。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

# 2026-11-08：gemini-3.8-flash と 3.7-flash は504（応答なし）が続いた。
# 3.5-flash を指定すると実際には 3.6-flash が応答していたため、再現性のため 3.6 を明示する
CHAT_MODEL = os.environ.get("DIY_CHAT_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.environ.get("DIY_EMBED_MODEL", "gemini-embedding-2")

# 有料で使った場合の料金（USD / 100万トークン）。評価の「1問あたりのAPI費用」の換算に使う。
# 2026-11-08時点の第三者の料金まとめによる値（Google公式の料金表では未確認）。未設定（None）なら「未設定」と表示する。
# gemini-3.6-flash：入力 $0.75、出力 $3.75（思考トークンも出力として課金）。gemini-embedding-2：$0.20。
# モデルを変えたら、ここも変える。
PRICE_CHAT_INPUT: float | None = 0.75
PRICE_CHAT_OUTPUT: float | None = 3.75
PRICE_EMBED: float | None = 0.20

# 方式3の思考（thinking）のトークン上限。0でオフ。調整はdevの質問だけで行う。
PARSE_THINKING_BUDGET = int(os.environ.get("DIY_PARSE_THINKING", "0"))
SELECT_THINKING_BUDGET = int(os.environ.get("DIY_SELECT_THINKING", "0"))
