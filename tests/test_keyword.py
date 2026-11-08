from diy_search.keyword import KeywordSearcher
from diy_search.models import Example, Size


def make(id_: str, title: str, purpose: str, description: str, actions: list[str]) -> Example:
    return Example(
        id=id_, title=title, category="c", purpose=purpose, actions=actions, location="室内",
        size_cm=Size(), max_load_kg=None, constraints=[], tools=[], description=description, materials=[],
    )


EXAMPLES = [
    make("A", "網戸の張り替え", "破れた網を新しくする", "古い網を外して新しい網を張る。", ["張る"]),
    make("B", "壁付けの棚", "壁に物を置く場所を作る", "L字金具で板を壁に留める。", ["載せる"]),
    make("C", "傘立て", "玄関で傘をまとめる", "木の板で箱を作る。", ["立てる"]),
]


def test_ranks_matching_example_first() -> None:
    results = KeywordSearcher(EXAMPLES).search("網戸が破れた")
    assert results[0].example_id == "A"


def test_returns_nothing_when_no_word_matches() -> None:
    assert KeywordSearcher(EXAMPLES).search("ぴょこぴょこ") == []


def test_respects_k() -> None:
    assert len(KeywordSearcher(EXAMPLES).search("板", k=1)) <= 1
