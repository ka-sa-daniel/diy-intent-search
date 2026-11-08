"""正解付けの補助。

- `uv run judge-sheet`：Excelで開ける判定シート（作例×質問の表）を private/judging/ に作る
- `uv run judge-import`：記入済みのシートを eval/judgments.csv（query_id, example_id, grade）に変換する

シートは作例が行、質問が列。セルに 2・1 を書く（空欄は0）。
"""

import csv
import sys

from diy_search.config import EVAL_DIR, ROOT
from diy_search.data import load_examples, load_queries

SHEET = ROOT / "private" / "judging" / "判定シート.csv"
INFO_COLS = ["id", "title", "場所", "幅×奥行×高さ(cm)", "耐荷重kg", "条件"]


def _size(s) -> str:
    return "×".join("-" if v is None else f"{v:g}" for v in (s.width, s.depth, s.height))


def make_sheet() -> None:
    if SHEET.exists():
        sys.exit(f"{SHEET} はすでにあります。上書きしないので、作り直すときは先に消してください。")
    queries = load_queries("dev") + load_queries("test")
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", encoding="utf-8-sig", newline="") as f:  # BOM付き：Excelで文字化けしない
        w = csv.writer(f)
        w.writerow(INFO_COLS + [q.id for q in queries])
        w.writerow(["", "（質問の型）", "", "", "", ""] + [q.type for q in queries])
        w.writerow(["", "（質問の本文）", "", "", "", ""] + [q.text for q in queries])
        for ex in load_examples():
            load = "-" if ex.max_load_kg is None else f"{ex.max_load_kg:g}"
            w.writerow([ex.id, ex.title, ex.location, _size(ex.size_cm), load, "、".join(ex.constraints)]
                       + [""] * len(queries))
    print(f"作成：{SHEET}")


def import_sheet() -> None:
    with SHEET.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[3:]
    query_ids = header[len(INFO_COLS):]
    out, errors = [], []
    for row in body:
        ex_id = row[0]
        for qid, cell in zip(query_ids, row[len(INFO_COLS):]):
            cell = cell.strip()
            if cell in ("", "0"):
                continue
            if cell not in ("1", "2"):
                errors.append(f"{ex_id}×{qid}: 「{cell}」は 0・1・2 ではありません")
                continue
            out.append((qid, ex_id, int(cell)))
    if errors:
        sys.exit("\n".join(errors))
    order = {qid: i for i, qid in enumerate(query_ids)}
    out.sort(key=lambda r: (order[r[0]], r[1]))
    path = EVAL_DIR / "judgments.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "example_id", "grade"])
        w.writerows(out)
    no_answer = [q for q in query_ids if q not in {r[0] for r in out}]
    print(f"出力：{path}（{len(out)}行）")
    print(f"grade 1以上が1件もない質問：{', '.join(no_answer) or 'なし'}")
