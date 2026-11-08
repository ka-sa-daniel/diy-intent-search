"""API使用量の記録と、有料で使った場合の料金への換算。"""

from dataclasses import dataclass

from diy_search import config


@dataclass
class Usage:
    chat_input_tokens: int = 0
    chat_output_tokens: int = 0  # 思考（thinking）のトークンも含める（出力として課金されるため）
    embed_tokens: int = 0

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            self.chat_input_tokens - other.chat_input_tokens,
            self.chat_output_tokens - other.chat_output_tokens,
            self.embed_tokens - other.embed_tokens,
        )

    def cost_usd(self) -> float | None:
        """料金表（config）が未設定なら None。"""
        prices = (config.PRICE_CHAT_INPUT, config.PRICE_CHAT_OUTPUT, config.PRICE_EMBED)
        if any(p is None for p in prices):
            return None if (self.chat_input_tokens or self.chat_output_tokens or self.embed_tokens) else 0.0
        return (
            self.chat_input_tokens * config.PRICE_CHAT_INPUT
            + self.chat_output_tokens * config.PRICE_CHAT_OUTPUT
            + self.embed_tokens * config.PRICE_EMBED
        ) / 1_000_000


class UsageMeter:
    """方式2・3が呼び出しごとに足し込む。評価では質問の前後の差を取る。"""

    def __init__(self) -> None:
        self.total = Usage()

    def snapshot(self) -> Usage:
        return Usage(**vars(self.total))
