"""設定。モデル名は環境変数で切り替えられる。APIキーはここで読まない（SDKが GEMINI_API_KEY を読む）。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

# 2026-11-08：gemini-3.8-flash と 3.7-flash は無料枠で504（応答なし）が続いたため、安定して動く3.5を使う
CHAT_MODEL = os.environ.get("DIY_CHAT_MODEL", "gemini-3.5-flash")
EMBED_MODEL = os.environ.get("DIY_EMBED_MODEL", "gemini-embedding-2")

# 有料で使った場合の料金（USD / 100万トークン）。評価の「1問あたりのAPI費用」の換算に使う。
# 2026-11-08時点の第三者の料金まとめによる値（Google公式の料金表では未確認）。未設定（None）なら「未設定」と表示する。
# gemini-3.5-flash：入力 $1.50、出力 $9.00（思考トークンも出力として課金）。gemini-embedding-2：$0.20。
# モデルを変えたら、ここも変える。
PRICE_CHAT_INPUT: float | None = 1.50
PRICE_CHAT_OUTPUT: float | None = 9.00
PRICE_EMBED: float | None = 0.20

# 方式3の思考（thinking）のトークン上限。0でオフ。調整はdevの質問だけで行う。
PARSE_THINKING_BUDGET = int(os.environ.get("DIY_PARSE_THINKING", "0"))
SELECT_THINKING_BUDGET = int(os.environ.get("DIY_SELECT_THINKING", "0"))
