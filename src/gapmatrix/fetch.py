"""Fetch paper text: OpenAlex abstract, open-access PDF, and optional manual text."""

import json
import re

import pandas as pd
import pymupdf
import requests

from .config import repo_path


def load_papers(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(repo_path(cfg["data"]["papers"]), dtype=str).fillna("")
    required = {"paper_id", "authors", "year", "title", "venue", "doi", "openalex_id",
                "use_openalex_abstract", "pdf_url"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"papers.csv is missing columns: {sorted(missing)}")
    return df


def rebuild_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, pos_list in inverted_index.items() for pos in pos_list]
    return " ".join(word for _, word in sorted(positions))


def fetch_openalex(paper: pd.Series, cfg: dict) -> dict:
    cache = repo_path(cfg["data"]["cache_dir"]) / f"{paper.paper_id}_openalex.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    base = cfg["fetch"]["openalex_url"]
    if paper.openalex_id:
        url = f"{base}/{paper.openalex_id}"
    elif paper.doi:
        url = f"{base}/doi:{paper.doi}"
    else:
        raise ValueError(f"{paper.paper_id}: needs a doi or openalex_id")

    resp = requests.get(url, timeout=cfg["fetch"]["timeout_seconds"])
    resp.raise_for_status()
    work = resp.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(work, ensure_ascii=False), encoding="utf-8")
    return work


def fetch_pdf_text(paper_id: str, pdf_url: str, cfg: dict) -> str:
    cache_dir = repo_path(cfg["data"]["cache_dir"])
    pdf_path = cache_dir / f"{paper_id}.pdf"
    if not pdf_path.exists():
        resp = requests.get(pdf_url, timeout=cfg["fetch"]["timeout_seconds"],
                            headers={"User-Agent": "Mozilla/5.0 (research script)"})
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise ValueError("response is not a PDF (likely a landing page or paywall)")
        pdf_path.write_bytes(resp.content)

    with pymupdf.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc][: cfg["fetch"]["max_pdf_pages"]]
    text = "\n".join(pages)
    # Drop everything after the reference list; it only adds retrieval noise.
    match = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    return text[: match.start()] if match else text


def collect_sources(paper: pd.Series, cfg: dict) -> list[dict]:
    """Return a list of {source, text} for one paper. Sources: abstract, pdf, manual."""
    sources = []
    work = fetch_openalex(paper, cfg)

    # Some OpenAlex abstracts are wrong (author lists, or another paper's abstract);
    # papers.csv marks those with use_openalex_abstract=no.
    abstract = rebuild_abstract(work.get("abstract_inverted_index"))
    if abstract and paper.use_openalex_abstract.lower() != "no":
        sources.append({"source": "abstract", "text": abstract})

    # A pdf_url in papers.csv (ACL Anthology or arXiv) overrides OpenAlex;
    # "none" skips a PDF that OpenAlex lists but that is not the paper itself.
    pdf_url = paper.pdf_url or ((work.get("best_oa_location") or {}).get("pdf_url")) or ""
    if pdf_url.lower() == "none":
        pdf_url = ""
    if cfg["fetch"]["use_open_access_pdf"] and pdf_url:
        try:
            sources.append({"source": "pdf", "text": fetch_pdf_text(paper.paper_id, pdf_url, cfg)})
        except Exception as e:  # noqa: BLE001 - a missing PDF must not stop the run
            print(f"  [{paper.paper_id}] PDF skipped: {e}")

    manual = repo_path(cfg["data"]["manual_dir"]) / f"{paper.paper_id}.txt"
    if manual.exists():
        sources.append({"source": "manual", "text": manual.read_text(encoding="utf-8")})

    return sources
