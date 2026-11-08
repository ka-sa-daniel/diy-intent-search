from diy_search import eval as ev
from diy_search.metrics import hit_at_k, top1
from diy_search.models import Query, Result
from diy_search.usage import Usage


def r(id_: str) -> Result:
    return Result(example_id=id_, title=id_, score=1.0)


JUDG = {("Q1", "B"): 1, ("Q1", "C"): 2}


def test_hit_and_top1() -> None:
    assert hit_at_k(JUDG, "Q1", [r("A"), r("B")])
    assert not top1(JUDG, "Q1", [r("A"), r("B")])
    assert top1(JUDG, "Q1", [r("C")])
    assert not hit_at_k(JUDG, "Q1", [r("A"), r("D")])


def test_unjudged_pairs_count_as_zero() -> None:
    assert not hit_at_k({}, "Q1", [r("B")])


def test_report_counts_no_match_only_for_agent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    q_ok = Query(id="Q1", type="purpose_only", text="x")
    q_nm = Query(id="Q2", type="no_match", text="y")
    runs = [
        ev.Run(q_ok, "keyword", [r("B")], 0.1, Usage()),
        ev.Run(q_nm, "keyword", [r("A")], 0.1, Usage()),
        ev.Run(q_ok, "agent", [r("A")], 1.0, Usage()),
        ev.Run(q_nm, "agent", [], 1.0, Usage()),
    ]
    text = open(ev.write_report(runs, JUDG, "dev"), encoding="utf-8").read()
    assert "| keyword | 1 | 100% | 100% | 対象外 |" in text
    assert "| agent | 1 | 0% | 0% | 100% |" in text
