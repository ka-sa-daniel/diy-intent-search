"""評価プログラム。`uv run eval [--set dev|test] [--methods keyword,vector,agent]`

全方式を同じ質問で実行し、results/YYYYMMDD-HHmm.md に表と外れた質問の一覧を書く。
実行の記録（返した作例・時間・トークン数）は同じ名前の .json に残す。
`uv run eval --rebuild results/XXXX.json` で、APIを呼ばずに表だけ作り直せる（集計方法を直したとき用）。
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from diy_search.config import RESULTS_DIR
from diy_search.data import load_examples, load_judgments, load_queries
from diy_search.agent import AgentSearcher
from diy_search.gemini import GeminiChat, GeminiEmbedder
from diy_search.keyword import KeywordSearcher
from diy_search.metrics import Judgments, grade, hit_at_k, rate, top1
from diy_search.models import Example, Query, Result, Searcher
from diy_search.usage import Usage, UsageMeter
from diy_search.vector import VectorSearcher

ALL_METHODS = ["keyword", "vector", "agent"]
TYPE_ORDER = ["purpose_only", "purpose_size", "problem_only", "keyword", "constraint", "vague", "no_match", "own"]


@dataclass
class Run:
    query: Query
    method: str
    results: list[Result]
    seconds: float
    usage: Usage
    error: str | None = None  # APIエラーで答えられなかったとき


def build_searchers(names: list[str], examples: list[Example], meter: UsageMeter) -> list[Searcher]:
    searchers: list[Searcher] = []
    for name in names:
        if name == "keyword":
            searchers.append(KeywordSearcher(examples))
        elif name == "vector":
            searchers.append(VectorSearcher(examples, GeminiEmbedder(meter)))
        elif name == "agent":
            vector = VectorSearcher(examples, GeminiEmbedder(meter))
            searchers.append(AgentSearcher(examples, vector, GeminiChat(meter)))
        else:
            raise SystemExit(f"不明な方式：{name}")
    return searchers


def run_all(searchers: list[Searcher], queries: list[Query], meter: UsageMeter) -> list[Run]:
    runs = []
    for s in searchers:
        for q in queries:
            before = meter.snapshot()
            t0 = time.perf_counter()
            error = None
            try:
                results = s.search(q.text, k=5)
            except Exception as e:  # 1問の失敗で全体を止めない。結果には「エラー」として残す
                results, error = [], f"{type(e).__name__}: {str(e)[:120]}"
            seconds = time.perf_counter() - t0
            runs.append(Run(q, s.name, results, seconds, meter.snapshot() - before, error))
            shown = error or [r.example_id for r in results]
            print(f"  {s.name:7s} {q.id}: {shown} ({seconds:.2f}s)")
    return runs


# ---------- 実行の記録 ----------

def save_runs(runs: list[Run], path: Path) -> None:
    data = [{"query": r.query.model_dump(), "method": r.method,
             "results": [x.model_dump() for x in r.results], "seconds": r.seconds,
             "usage": asdict(r.usage), "error": r.error} for r in runs]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_runs(path: Path) -> list[Run]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Run(Query(**d["query"]), d["method"], [Result(**x) for x in d["results"]],
                d["seconds"], Usage(**d["usage"]), d["error"]) for d in data]


# ---------- 集計 ----------

def expects_no_match(q: Query, judgments: Judgments) -> bool:
    """「該当なし」が正しい答えの問題か。no_match型と、grade 1以上の作例が1件もない自作の質問。"""
    if q.type == "no_match":
        return True
    return q.type == "own" and not any(qid == q.id and g >= 1 for (qid, _), g in judgments.items())


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def _summary_row(method: str, runs: list[Run], judgments: Judgments) -> str:
    answerable = [r for r in runs if not expects_no_match(r.query, judgments)]
    no_match = [r for r in runs if expects_no_match(r.query, judgments)]
    hit = rate([hit_at_k(judgments, r.query.id, r.results) for r in answerable])
    t1 = rate([top1(judgments, r.query.id, r.results) for r in answerable])
    # 「該当なし」の正答率は方式3だけが対象（空のリスト＝該当なし）
    nm = rate([not r.results for r in no_match]) if method == "agent" else None
    avg_sec = sum(r.seconds for r in runs) / len(runs)
    costs = [r.usage.cost_usd() for r in runs]
    cost = "未設定" if any(c is None for c in costs) else f"${sum(costs) / len(costs):.5f}"
    nm_cell = "対象外" if method != "agent" else f"{_pct(nm)}（{sum(not r.results for r in no_match)}/{len(no_match)}）" if no_match else "-"
    return f"| {method} | {len(answerable)} | {_pct(hit)} | {_pct(t1)} | {nm_cell} | {avg_sec:.2f}s | {cost} |"


def _summary_table(by_method: dict[str, list[Run]], judgments: Judgments) -> list[str]:
    lines = [
        "| 方式 | 当たりがある問題数 | Hit@5 | Top1 | 該当なしの正答率 | 平均応答時間 | 1問あたりの費用 |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, runs in by_method.items():
        if runs:
            lines.append(_summary_row(method, runs, judgments))
    return lines


def _type_table(by_method: dict[str, list[Run]], judgments: Judgments) -> list[str]:
    methods = list(by_method)
    lines = [
        "| 質問の型 | 問題数 | " + " | ".join(f"{m} Hit@5 / Top1" for m in methods) + " |",
        "|---|---|" + "---|" * len(methods),
    ]
    types = [t for t in TYPE_ORDER if t not in ("no_match", "own")]
    for t in types:
        cells, n = [], 0
        for m in methods:
            rs = [r for r in by_method[m] if r.query.type == t]
            n = len(rs)
            hit = rate([hit_at_k(judgments, r.query.id, r.results) for r in rs])
            t1 = rate([top1(judgments, r.query.id, r.results) for r in rs])
            cells.append(f"{_pct(hit)} / {_pct(t1)}")
        if n:
            lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")
    return lines


def _fmt_results(q: Query, results: list[Result], judgments: Judgments, error: str | None = None) -> str:
    if error:
        return f"（エラー：{error}）"
    if not results:
        return "（該当なし）"
    return "、".join(f"{r.example_id}[{grade(judgments, q.id, r.example_id)}]" for r in results)


def _misses(by_method: dict[str, list[Run]], judgments: Judgments) -> list[str]:
    lines = ["| 方式 | 質問 | 型 | 本文 | 返した作例[grade] |", "|---|---|---|---|---|"]
    for method, runs in by_method.items():
        for r in runs:
            if expects_no_match(r.query, judgments):
                missed = method == "agent" and bool(r.results)
            else:
                missed = not hit_at_k(judgments, r.query.id, r.results)
            if missed:
                lines.append(f"| {method} | {r.query.id} | {r.query.type} | {r.query.text} | {_fmt_results(r.query, r.results, judgments, r.error)} |")
    return lines if len(lines) > 2 else ["外れた質問はありません。"]


def write_report(runs: list[Run], judgments: Judgments, query_set: str,
                 now: datetime | None = None, note: str | None = None) -> str:
    now = now or datetime.now()
    main_runs = [r for r in runs if r.query.type != "own"]
    own_runs = [r for r in runs if r.query.type == "own"]

    def group(rs: list[Run]) -> dict[str, list[Run]]:
        g: dict[str, list[Run]] = defaultdict(list)
        for r in rs:
            g[r.method].append(r)
        return dict(g)

    out = [f"# 評価結果 {now:%Y-%m-%d %H:%M}", "", f"- 質問：`queries_{query_set}.json`", "- 当たり：grade 1以上（2＝そのまま使える、1＝寸法や材料を変えれば参考になる）",
           "- 該当なしの問題：`no_match` 型と、grade 1以上の作例がない自作の質問。Hit@5・Top1の分母から外し、方式3が「該当なし」と返せたかだけを見る", ""]
    if note:
        out += [f"> {note}", ""]
    if not judgments:
        out += ["> **注意**：`eval/judgments.csv` がないため、すべて grade 0 として計算しています。指標は意味を持ちません。", ""]
    if main_runs:
        g = group(main_runs)
        out += ["## 方式ごとの結果（LLMで生成した質問）", "", *_summary_table(g, judgments), "",
                "## 質問の型ごとの結果", "", *_type_table(g, judgments), "",
                "## 外れた質問", "", *_misses(g, judgments), ""]
    if own_runs:
        g = group(own_runs)
        out += ["## 自分で書いた質問（別集計）", "", *_summary_table(g, judgments), "",
                "### 外れた質問", "", *_misses(g, judgments), ""]
    out += ["## 全結果", "", "| 方式 | 質問 | 返した作例[grade] | 時間 |", "|---|---|---|---|"]
    out += [f"| {r.method} | {r.query.id} | {_fmt_results(r.query, r.results, judgments, r.error)} | {r.seconds:.2f}s |" for r in runs]
    errors = [r for r in runs if r.error]
    if errors:
        out.insert(4, f"> **注意**：{len(errors)}件がAPIエラーで答えられず、外れとして数えています（全結果の表を参照）。\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{now:%Y%m%d-%H%M}{'' if query_set == 'test' else '-' + query_set}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    save_runs(runs, path.with_suffix(".json"))
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="3方式を同じ質問で評価する")
    parser.add_argument("--set", choices=["dev", "test"], default="test")
    parser.add_argument("--methods", default=",".join(ALL_METHODS), help="例：keyword,vector")
    parser.add_argument("--rebuild", type=Path, help="保存した実行記録（.json）から、APIを呼ばずに表だけ作り直す")
    args = parser.parse_args()

    if args.rebuild:
        runs = load_runs(args.rebuild)
        stem = args.rebuild.stem
        query_set = "dev" if stem.endswith("-dev") else "test"
        now = datetime.strptime(stem[:13], "%Y%m%d-%H%M")
        note = f"実行した時の記録から、{datetime.now():%Y-%m-%d %H:%M} に表だけ作り直しました（APIは呼び直していません）。"
        print("出力：", write_report(runs, load_judgments(), query_set, now=now, note=note))
        return

    examples = load_examples()
    queries = load_queries(args.set)
    judgments = load_judgments()
    if not judgments:
        print("[warn] eval/judgments.csv がありません。指標は意味を持ちません。")

    meter = UsageMeter()
    searchers = build_searchers(args.methods.split(","), examples, meter)
    runs = run_all(searchers, queries, meter)
    print("出力：", write_report(runs, judgments, args.set))
