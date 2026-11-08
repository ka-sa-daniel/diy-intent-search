"""評価指標。grade 1以上を「当たり」とする。"""

from diy_search.models import Result

Judgments = dict[tuple[str, str], int]


def grade(judgments: Judgments, query_id: str, example_id: str) -> int:
    return judgments.get((query_id, example_id), 0)


def hit_at_k(judgments: Judgments, query_id: str, results: list[Result], k: int = 5) -> bool:
    return any(grade(judgments, query_id, r.example_id) >= 1 for r in results[:k])


def top1(judgments: Judgments, query_id: str, results: list[Result]) -> bool:
    return bool(results) and grade(judgments, query_id, results[0].example_id) >= 1


def rate(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None
