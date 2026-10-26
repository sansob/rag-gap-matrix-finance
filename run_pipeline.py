"""Entry point: fetch -> index -> generate (+ export).

Examples:
    python run_pipeline.py all
    python run_pipeline.py generate --dry-run          # build prompts, no Groq calls
    python run_pipeline.py generate --only P01         # test on one paper first
    python run_pipeline.py review                      # merge data/review/verdicts.csv
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gapmatrix.config import load_config, repo_path, set_seed  # noqa: E402
from gapmatrix.fetch import collect_sources, load_papers  # noqa: E402


def step_fetch(papers, cfg) -> None:
    out = {}
    for p in papers.itertuples():
        sources = collect_sources(p, cfg)
        out[p.paper_id] = sources
        print(f"  [{p.paper_id}] " + ", ".join(f"{s['source']}={len(s['text'])} chars" for s in sources))
        if not sources:
            print(f"  [{p.paper_id}] WARNING: no text found; add data/manual/{p.paper_id}.txt")
    path = repo_path(cfg["data"]["cache_dir"]) / "sources.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def step_index(cfg, rebuild: bool) -> None:
    from gapmatrix.index import build_index

    sources = json.loads((repo_path(cfg["data"]["cache_dir"]) / "sources.json").read_text(encoding="utf-8"))
    print(f"  added {build_index(sources, cfg, rebuild=rebuild)} new chunks")


def step_generate(papers, cfg, dry_run: bool) -> None:
    from gapmatrix.export import export
    from gapmatrix.generate import generate_matrix

    rows = generate_matrix(papers, cfg, dry_run=dry_run)
    if dry_run:
        print("  dry run: prompts written to results/llm_cache/*_prompt.txt")
        return
    run_info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["seed"],
        "embedding_model": cfg["embedding"]["model"],
        "generation_model": cfg["generation"]["model"],
        "top_k": cfg["retrieval"]["top_k"],
        "chunking": cfg["chunking"],
        "papers": [r["paper"].paper_id for r in rows],
    }
    print(f"  cell checks: {export(rows, cfg, run_info)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG pipeline for a literature gap matrix")
    ap.add_argument("step", choices=["fetch", "index", "generate", "review", "all"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="+", help="paper_id(s) to generate (index always uses all)")
    ap.add_argument("--dry-run", action="store_true", help="build prompts without calling Groq")
    ap.add_argument("--rebuild", action="store_true", help="re-embed every paper (after chunking/model changes)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    papers = load_papers(cfg)

    if args.step in ("fetch", "all"):
        print("[fetch]")
        step_fetch(papers, cfg)
    if args.step in ("index", "all"):
        print("[index]")
        step_index(cfg, args.rebuild)
    if args.step in ("generate", "all"):
        print("[generate]")
        subset = papers[papers.paper_id.isin(args.only)] if args.only else papers
        step_generate(subset, cfg, args.dry_run)
    if args.step == "review":
        from gapmatrix.export import apply_review

        print("[review]")
        print(json.dumps(apply_review(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
