"""方式3：エージェント（手順が決まったパイプライン。ツール呼び出し型ではない）。

1. LLMで質問を項目に分ける（構造化出力）
2. 外せない条件だけで除外する（寸法では除外しない）
3. 1で得た目的でベクトル検索し、上位10件を取る
4. 寸法・耐荷重が桁違い（3倍以上）のものだけ順位を下げる
5. 上位5件からLLMが参考になるものを選び、理由と寸法の違いを書く。なければ「該当なし」（空のリスト）
"""

import json
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from diy_search.config import PARSE_THINKING_BUDGET, SELECT_THINKING_BUDGET
from diy_search.models import Example, Result, Size
from diy_search.vector import VectorSearcher

HardConstraint = Literal["穴をあけられない", "賃貸", "屋外で使う", "電動工具を使えない"]


class ParsedQuery(BaseModel):
    purpose: str = Field(description="何をしたいか・何を解決したいか。DIYの作例を探すための短い文")
    actions: list[str] = Field(description="動作（載せる、掛ける、吊るす、仕切る、隠す、ふさぐ など）")
    location: str | None = Field(description="場所（室内、玄関、ベランダ、屋外、壁面、キッチン など）。不明ならnull")
    size_cm: Size = Field(description="質問に書かれた寸法（cm）。書かれていない項目はnull")
    load_kg: float | None = Field(description="載せる・掛ける物の重さ（kg）。書かれていなければnull")
    constraints: list[HardConstraint] = Field(
        description="質問にはっきり書かれた、外せない条件だけ。推測で足さない")
    missing: list[str] = Field(description="作例を選ぶのに足りない情報（聞き返すなら何を聞くか）")


class Pick(BaseModel):
    example_id: str
    reason: str = Field(description="この作例が参考になる理由（1文）")
    size_note: str = Field(description="寸法・耐荷重・条件の違いと、どう変えればよいか（1文）。違いがなければ「そのまま使える」")


class Selection(BaseModel):
    picks: list[Pick] = Field(description="参考になる作例を、役に立つ順に。なければ空")
    no_match_reason: str | None = Field(description="picksが空のとき、該当なしの理由。そうでなければnull")


class Chat(Protocol):
    def generate(self, prompt: str, schema: type, thinking_budget: int = 0): ...


PARSE_PROMPT = """あなたはホームセンターのDIY相談の担当者です。お客さんの相談文から、DIYの作例を探すための項目を取り出してください。
- purpose は、作例を探しやすい言葉で短く書く（お客さんの言葉をそのまま写さなくてよい）。
- constraints は、相談文にはっきり書かれた条件だけを入れる。推測で足さない。
  - 「穴をあけられない」：壁や天井に穴をあけたくない・あけられない
  - 「賃貸」：賃貸住まいと書かれている
  - 「屋外で使う」：雨の当たる所・屋外・ベランダで使う
  - 「電動工具を使えない」：工具がない・電動工具がない
- 寸法は cm、重さは kg に直す。

相談文：{query}"""

SELECT_PROMPT = """あなたはホームセンターのDIY相談の担当者です。お客さんの相談に対して、候補の作例から参考になるものを選んでください。
- 寸法が違っても、作り方や材料を変えれば参考になるなら選んでよい。その場合は size_note に違いと変え方を書く。
- 目的が違う作例は選ばない。参考になるものがなければ picks を空にし、no_match_reason に理由を書く。
- 候補にない id は使わない。

相談文：{query}
読み取った要望：{parsed}

候補の作例：
{candidates}"""


# ---------- 2. 外せない条件での除外 ----------

def _uses_power_tool(ex: Example) -> bool:
    return any(w in t for t in ex.tools for w in ("電動", "丸のこ", "ドリル"))


def violates(ex: Example, constraint: str) -> bool:
    """作例がその条件を満たせないなら True。"""
    if constraint == "穴をあけられない":
        return "穴あけ必要" in ex.constraints
    if constraint == "賃貸":
        return "賃貸可" not in ex.constraints
    if constraint == "屋外で使う":
        return "屋外可" not in ex.constraints
    if constraint == "電動工具を使えない":
        return _uses_power_tool(ex)
    return False


def allowed_ids(examples: list[Example], constraints: list[str]) -> set[str]:
    return {ex.id for ex in examples if not any(violates(ex, c) for c in constraints)}


# ---------- 4. 桁違いだけ順位を下げる ----------

RATIO_LIMIT = 3.0


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a <= 0 or b <= 0:
        return None
    return max(a, b) / min(a, b)


def scale_mismatch(ex: Example, size: Size, load_kg: float | None) -> bool:
    """寸法のどれかが3倍以上違う、または求める重さが作例の耐荷重の3倍以上なら True。"""
    for dim in ("width", "depth", "height"):
        r = _ratio(getattr(size, dim), getattr(ex.size_cm, dim))
        if r is not None and r >= RATIO_LIMIT:
            return True
    if load_kg is not None and ex.max_load_kg is not None and load_kg >= ex.max_load_kg * RATIO_LIMIT:
        return True
    return False


def demote_mismatches(results: list[Result], by_id: dict[str, Example], size: Size, load_kg: float | None) -> list[Result]:
    """順番は保ったまま、桁違いのものだけ後ろへ回す（安定ソート）。"""
    return sorted(results, key=lambda r: scale_mismatch(by_id[r.example_id], size, load_kg))


# ---------- 本体 ----------

def _candidate_text(ex: Example) -> str:
    s = ex.size_cm
    return json.dumps({
        "id": ex.id, "title": ex.title, "purpose": ex.purpose, "location": ex.location,
        "size_cm": {"width": s.width, "depth": s.depth, "height": s.height},
        "max_load_kg": ex.max_load_kg, "constraints": ex.constraints,
        "materials": [m.name for m in ex.materials],
    }, ensure_ascii=False)


class AgentSearcher:
    name = "agent"

    def __init__(self, examples: list[Example], vector: VectorSearcher, chat: Chat):
        self.examples = examples
        self.by_id = {ex.id: ex for ex in examples}
        self.vector = vector
        self.chat = chat
        self.last_parsed: ParsedQuery | None = None
        self.last_no_match_reason: str | None = None

    def search(self, query: str, k: int = 5) -> list[Result]:
        # 1. 項目に分ける
        parsed: ParsedQuery = self.chat.generate(
            PARSE_PROMPT.format(query=query), ParsedQuery, PARSE_THINKING_BUDGET)
        self.last_parsed = parsed
        # 2. 外せない条件で除外
        allowed = allowed_ids(self.examples, parsed.constraints)
        # 3. 目的でベクトル検索（上位10件）
        candidates = self.vector.search(parsed.purpose, k=10, allowed_ids=allowed)
        # 4. 桁違いだけ下げる
        candidates = demote_mismatches(candidates, self.by_id, parsed.size_cm, parsed.load_kg)[:k]
        if not candidates:
            self.last_no_match_reason = "条件に合う作例がありません"
            return []
        # 5. LLMが選んで理由を書く
        selection: Selection = self.chat.generate(
            SELECT_PROMPT.format(
                query=query,
                parsed=parsed.model_dump_json(exclude={"missing"}),
                candidates="\n".join(_candidate_text(self.by_id[c.example_id]) for c in candidates),
            ),
            Selection, SELECT_THINKING_BUDGET)
        self.last_no_match_reason = selection.no_match_reason
        score = {c.example_id: c.score for c in candidates}
        results, seen = [], set()
        for p in selection.picks:
            if p.example_id not in score or p.example_id in seen:  # 候補外のidは捨てる
                continue
            seen.add(p.example_id)
            ex = self.by_id[p.example_id]
            results.append(Result(example_id=ex.id, title=ex.title, score=score[ex.id],
                                  reason=p.reason, size_note=p.size_note))
        return results[:k]
