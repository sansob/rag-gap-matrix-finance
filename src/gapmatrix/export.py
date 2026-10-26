"""Write the gap matrix to CSV, XLSX, and Markdown, plus an evidence file for review.

Two outputs:
- gap_matrix.*           raw LLM output after the automatic quote check
- gap_matrix_reviewed.*  the same matrix after cell-by-cell review (data/review/verdicts.csv)
"""

import json

import pandas as pd

from .config import repo_path


def paper_label(p) -> str:
    return f"{p.authors} ({p.year}). {p.title}. {p.venue}. https://doi.org/{p.doi}" if p.doi \
        else f"{p.authors} ({p.year}). {p.title}. {p.venue}."


def write_tables(df: pd.DataFrame, ev: pd.DataFrame, base) -> None:
    df.to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
    ev.to_csv(f"{base}_evidence.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(f"{base}.xlsx") as xw:
        df.to_excel(xw, sheet_name="gap_matrix", index=False)
        ev.to_excel(xw, sheet_name="evidence", index=False)
    with open(f"{base}.md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))


def export(rows: list[dict], cfg: dict, run_info: dict) -> dict:
    out_dir = repo_path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [c["name"] for c in cfg["columns"]]

    matrix, evidence = [], []
    for r in rows:
        p = r["paper"]
        matrix.append({"ID": p.paper_id, "Paper": paper_label(p),
                       **{n: r["cells"][n]["content"] for n in names}})
        for n in names:
            cell = r["cells"][n]
            evidence.append({"paper_id": p.paper_id, "column": n, "check": cell["check"],
                             "content": cell["content"], "quote": cell["quote"],
                             "evidence_ids": "; ".join(cell["evidence_ids"])})

    df, ev = pd.DataFrame(matrix), pd.DataFrame(evidence)
    write_tables(df, ev, out_dir / cfg["output"]["basename"])

    summary = ev["check"].value_counts().to_dict()
    run_info = {**run_info, "cell_checks": summary}
    (out_dir / "run_log.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def apply_review(cfg: dict) -> dict:
    """Merge review verdicts into the matrix and compare them with the automatic check.

    verdicts.csv columns: paper_id, column, verdict, reason, corrected_content, support_ids.
    PARTIAL / UNSUPPORTED / NOT_STATED_WRONG cells are replaced by corrected_content.
    """
    out_dir = repo_path(cfg["output"]["dir"])
    base = out_dir / cfg["output"]["basename"]
    df = pd.read_csv(f"{base}.csv", dtype=str).fillna("")
    ev = pd.read_csv(f"{base}_evidence.csv", dtype=str).fillna("")
    rv = pd.read_csv(repo_path(cfg["review"]["verdicts"]), dtype=str).fillna("")

    ev = ev.merge(rv, on=["paper_id", "column"], how="left", validate="one_to_one")
    missing = ev[ev["verdict"].isna() | (ev["verdict"] == "")]
    if len(missing):
        raise ValueError(f"{len(missing)} cells have no review verdict, e.g. "
                         f"{missing[['paper_id', 'column']].head(3).values.tolist()}")

    fix = ev["verdict"].isin(["PARTIAL", "UNSUPPORTED", "NOT_STATED_WRONG"])
    ev["final_content"] = ev["content"].where(~fix, ev["corrected_content"])
    for r in ev[fix].itertuples():
        df.loc[df["ID"] == r.paper_id, r.column] = r.final_content

    write_tables(df, ev, f"{base}_reviewed")
    agreement = pd.crosstab(ev["check"], ev["verdict"])
    agreement.to_csv(out_dir / "review_vs_auto_check.csv", encoding="utf-8-sig")
    summary = {"verdicts": ev["verdict"].value_counts().to_dict(),
               "cells_corrected": int(fix.sum()), "cells_total": len(ev),
               "auto_check_vs_review": agreement.to_dict()}
    (out_dir / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
    return summary
