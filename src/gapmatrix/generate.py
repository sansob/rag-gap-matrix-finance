"""Generation with Groq: one call per paper, with per-column retrieved context.

Groq's free tier is rate limited, so this module:
- makes ONE call per paper (all columns at once), not one call per cell;
- caches every response on disk keyed by a hash of the prompt, so re-runs are free;
- supports a dry run that builds prompts without calling the API.
"""

import difflib
import hashlib
import json
import re
import time

from openai import OpenAI, RateLimitError

from .config import groq_api_key, repo_path
from .index import retrieve

SYSTEM_PROMPT = """You are a careful research assistant building a literature gap matrix.
You only use the CONTEXT chunks you are given. You never use outside knowledge about the paper.
If the context does not support a column, write exactly the not-stated label for that column,
leave evidence_ids empty, and leave quote empty. Return valid JSON only."""


def build_prompt(paper, cfg: dict) -> tuple[str, dict[str, list[dict]]]:
    retrieved = {col["name"]: retrieve(paper.paper_id, col["query"], cfg) for col in cfg["columns"]}

    # Deduplicate chunks across columns so each chunk is sent once.
    chunks = {}
    for hits in retrieved.values():
        for h in hits:
            chunks.setdefault(h["id"], h["text"])
    context = "\n\n".join(f"[{cid}]\n{text}" for cid, text in chunks.items())

    column_spec = "\n".join(
        f'- "{col["name"]}": {col["instruction"].strip()} '
        f'Most relevant chunks: {", ".join(h["id"] for h in retrieved[col["name"]])}'
        for col in cfg["columns"]
    )
    gen = cfg["generation"]
    prompt = f"""PAPER: {paper.title} ({paper.authors}, {paper.year})

RESEARCH FOCUS (for the Gap column only):
{cfg["research_focus"].strip()}

COLUMNS TO FILL:
{column_spec}

RULES:
- Write each "content" in {gen["output_language"]}, 1-3 concise sentences. Keep technical terms in English.
- Not-stated label: "{gen["not_stated_label"]}".
- "evidence_ids": the chunk ids that support the content.
- "quote": one short span (max 25 words) copied character-for-character from one of those
  chunks, in the chunk's ORIGINAL language (usually English). Do NOT translate or paraphrase
  the quote, even though "content" is in {gen["output_language"]}. One contiguous span, no "...".
- Every number in "content" must appear in the cited chunks.
- The Gap column is an analytic judgement, not a fact lookup: always fill it by comparing what
  the paper covers (or states as a limitation) with the research focus, and quote the chunk
  that shows what the paper covers or lacks.
- Use plain ASCII hyphens (-) in your text.

OUTPUT FORMAT (JSON object, one key per column):
{{"<column name>": {{"content": "...", "evidence_ids": ["..."], "quote": "..."}}}}

CONTEXT:
{context}"""
    return prompt, retrieved


def _normalize(s: str) -> str:
    # Compare letters and digits only: PDF text breaks words across lines ("task-\naware")
    # and mixes dash/quote variants, which would fail an exact match on a correct quote.
    return re.sub(r"[^0-9a-z]", "", s.lower())


def quote_match(quote: str, chunks: list[str]) -> float:
    """Share of the quote found in the cited chunks: 1.0 = every segment matches exactly.

    The LLM often joins segments with "..." or changes a word, so each segment is matched
    on its own; a segment with no exact match gets its best fuzzy score
    (difflib matching blocks against the chunk).
    """
    segments = [_normalize(s) for s in re.split(r"\.\.\.|…", quote)]
    segments = [s for s in segments if len(s) >= 15]
    if not segments:
        return 0.0
    haystacks = [_normalize(c) for c in chunks]
    scores = []
    for seg in segments:
        if any(seg in h for h in haystacks):
            scores.append(1.0)
            continue
        best = 0.0
        for h in haystacks:
            blocks = difflib.SequenceMatcher(None, seg, h, autojunk=False).get_matching_blocks()
            best = max(best, sum(b.size for b in blocks if b.size >= 4) / len(seg))
        scores.append(best)
    return min(scores)


def check_cells(cells: dict, chunk_text: dict, cfg: dict) -> dict:
    """Guardrail: evidence ids must exist and the quote must appear in a cited chunk."""
    label = cfg["generation"]["not_stated_label"]
    checked = {}
    for col in cfg["columns"]:
        name = col["name"]
        cell = cells.get(name) or {}
        content = re.sub(r"[‐-―]", "-", (cell.get("content") or "").strip()) or label
        ids = [i for i in cell.get("evidence_ids") or [] if i in chunk_text]
        quote = (cell.get("quote") or "").strip()
        if content == label:
            status = "not_stated"
        elif not ids:
            status = "no_evidence"
        else:
            score = quote_match(quote, [chunk_text[i] for i in ids])
            status = ("quote_verified" if score == 1.0
                      else "quote_fuzzy" if score >= cfg["generation"]["fuzzy_quote_threshold"]
                      else "quote_not_found")
        checked[name] = {"content": content, "evidence_ids": ids, "quote": quote, "check": status}
    return checked


def call_groq(client: OpenAI, prompt: str, cfg: dict) -> tuple[dict, dict]:
    gen = cfg["generation"]
    kwargs = dict(
        model=gen["model"],
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=gen["temperature"],
        max_tokens=gen["max_tokens"],
        seed=cfg["seed"],
        response_format={"type": "json_object"},
    )
    if gen.get("reasoning_effort"):
        kwargs["reasoning_effort"] = gen["reasoning_effort"]

    for attempt in range(1, gen["max_retries"] + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            usage = resp.usage.model_dump() if resp.usage else {}
            return json.loads(resp.choices[0].message.content), usage
        except RateLimitError:
            wait = 15 * attempt
            print(f"  rate limited, waiting {wait}s (attempt {attempt}/{gen['max_retries']})")
            time.sleep(wait)
        except json.JSONDecodeError:
            print(f"  invalid JSON, retrying (attempt {attempt}/{gen['max_retries']})")
    raise RuntimeError("Groq call failed after all retries")


def generate_matrix(papers, cfg: dict, dry_run: bool = False) -> list[dict]:
    cache_dir = repo_path(cfg["output"]["dir"]) / "llm_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = None if dry_run else OpenAI(api_key=groq_api_key(), base_url=cfg["generation"]["base_url"])

    rows, total_usage = [], {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}
    for paper in papers.itertuples():
        prompt, retrieved = build_prompt(paper, cfg)
        chunk_text = {h["id"]: h["text"] for hits in retrieved.values() for h in hits}
        key = hashlib.sha256((cfg["generation"]["model"] + prompt).encode()).hexdigest()[:16]
        cache = cache_dir / f"{paper.paper_id}_{key}.json"

        if dry_run:
            print(f"--- {paper.paper_id}: ~{len(prompt) // 4} prompt tokens, {len(chunk_text)} chunks")
            (cache_dir / f"{paper.paper_id}_prompt.txt").write_text(prompt, encoding="utf-8")
            continue

        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            print(f"  [{paper.paper_id}] cached")
        else:
            print(f"  [{paper.paper_id}] calling Groq")
            cells, usage = call_groq(client, prompt, cfg)
            record = {"paper_id": paper.paper_id, "model": cfg["generation"]["model"],
                      "seed": cfg["seed"], "usage": usage, "raw": cells, "retrieved": retrieved}
            cache.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["api_calls"] += 1
            time.sleep(cfg["generation"]["delay_seconds"])

        cited = {h["id"]: h["text"] for hits in record["retrieved"].values() for h in hits}
        rows.append({"paper": paper, "cells": check_cells(record["raw"], cited, cfg)})

    print(f"Groq usage this run: {total_usage}")
    return rows
