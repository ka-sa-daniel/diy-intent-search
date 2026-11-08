from diy_search.agent import (
    AgentSearcher, ParsedQuery, Pick, Selection, allowed_ids, demote_mismatches, scale_mismatch, violates,
)
from diy_search.models import Example, Result, Size


def make(id_: str, title: str, constraints=(), tools=(), size=Size(), load=None) -> Example:
    return Example(
        id=id_, title=title, category="c", purpose=title, actions=[], location="室内",
        size_cm=size, max_load_kg=load, constraints=list(constraints), tools=list(tools),
        description="", materials=[],
    )


def test_no_drilling_excludes_only_drilling_examples() -> None:
    drill = make("A", "壁付け棚", constraints=["穴あけ必要"])
    stand = make("B", "置く棚", constraints=["壁に穴をあけない"])
    assert violates(drill, "穴をあけられない")
    assert not violates(stand, "穴をあけられない")


def test_rental_requires_rental_ok() -> None:
    assert allowed_ids([make("A", "x", ["賃貸可"]), make("B", "y")], ["賃貸"]) == {"A"}


def test_power_tool_constraint() -> None:
    assert violates(make("A", "x", tools=["電動ドリルドライバー"]), "電動工具を使えない")
    assert not violates(make("B", "y", tools=["メジャー"]), "電動工具を使えない")


def test_scale_mismatch_uses_three_times_rule() -> None:
    ex = make("A", "棚", size=Size(width=60), load=8)
    assert not scale_mismatch(ex, Size(width=90), None)  # 1.5倍は下げない
    assert scale_mismatch(ex, Size(width=200), None)  # 3倍以上は下げる
    assert scale_mismatch(ex, Size(), 30)  # 耐荷重8kgに30kg
    assert not scale_mismatch(ex, Size(height=100), None)  # 作例側がnullの寸法は比べない


def test_demote_keeps_order_otherwise() -> None:
    by_id = {"A": make("A", "a", size=Size(width=10)), "B": make("B", "b", size=Size(width=60)),
             "C": make("C", "c", size=Size(width=70))}
    rs = [Result(example_id=i, title=i, score=1.0) for i in "ABC"]
    assert [r.example_id for r in demote_mismatches(rs, by_id, Size(width=60), None)] == ["B", "C", "A"]


class FakeVector:
    def search(self, query, k=5, allowed_ids=None):
        ids = [i for i in ["A", "B", "C"] if allowed_ids is None or i in allowed_ids]
        return [Result(example_id=i, title=i, score=1.0 - n * 0.1) for n, i in enumerate(ids)][:k]


class FakeChat:
    def __init__(self, parsed: ParsedQuery, selection: Selection):
        self.replies = [parsed, selection]
        self.prompts = []

    def generate(self, prompt, schema, thinking_budget=0):
        self.prompts.append(prompt)
        return self.replies.pop(0)


EXAMPLES = [make("A", "壁付け棚", ["穴あけ必要"]), make("B", "置く棚", ["賃貸可"]), make("C", "傘立て", ["賃貸可"])]


def parsed(constraints=()) -> ParsedQuery:
    return ParsedQuery(purpose="棚", actions=[], location=None, size_cm=Size(), load_kg=None,
                       constraints=list(constraints), missing=[])


def test_agent_filters_and_ignores_unknown_ids() -> None:
    sel = Selection(picks=[Pick(example_id="Z", reason="", size_note=""),
                           Pick(example_id="B", reason="置ける", size_note="そのまま使える")],
                    no_match_reason=None)
    chat = FakeChat(parsed(["穴をあけられない"]), sel)
    results = AgentSearcher(EXAMPLES, FakeVector(), chat).search("賃貸で棚がほしい")
    assert [r.example_id for r in results] == ["B"]
    assert results[0].reason == "置ける"
    assert '"id": "A"' not in chat.prompts[1]  # 穴あけ必要の作例はLLMに渡らない


def test_agent_returns_empty_for_no_match() -> None:
    sel = Selection(picks=[], no_match_reason="DIYの作例ではない")
    s = AgentSearcher(EXAMPLES, FakeVector(), FakeChat(parsed(), sel))
    assert s.search("車を直したい") == []
    assert s.last_no_match_reason == "DIYの作例ではない"


def test_outdoor_is_not_a_filter() -> None:
    # 室内用でも、材料を変えれば屋外で参考になるので除外しない
    assert not violates(make("A", "室内の棚"), "屋外で使う")
