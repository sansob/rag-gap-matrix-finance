# RAG Gap Matrix: Retrieval-Augmented Generation in Finance

A small RAG pipeline that builds a **literature gap matrix** for research on RAG in the
finance and banking domain. It retrieves evidence from each paper's abstract (and its
open-access PDF when one exists), then asks an LLM on Groq to fill each matrix cell
using only that evidence. Every generated cell is then reviewed against its source chunks.

Built for the RTM 2 assignment of *Introduction to Generative AI*, Master of Artificial
Intelligence, Universitas Dian Nuswantoro.

## Pipeline

```
data/papers.csv ─► fetch ─► index ─► generate ─► review ─► results/
 (13 verified     OpenAlex  chunk +   Groq LLM    verdicts   gap_matrix.*           raw LLM output
  papers)         abstract  local     (JSON per   merged     gap_matrix_reviewed.*  after review
                  + OA PDF  embedding paper)                 review_summary.json
                            + Chroma
```

1. **fetch**: pulls each paper's abstract from the [OpenAlex API](https://docs.openalex.org/)
   (`abstract_inverted_index`) and, when available, its open-access PDF. PDF text is read with
   PyMuPDF and cut at the reference list. Extra text can be added by hand in
   `data/manual/<paper_id>.txt`.
2. **index**: splits each paper's text into fixed-size character chunks with overlap,
   embeds them **locally on CPU** with `paraphrase-multilingual-mpnet-base-v2`, and stores
   them in a persistent Chroma collection. The model runs on ONNX Runtime through
   [fastembed](https://github.com/qdrant/fastembed), so no torch or GPU is needed. Indexing is
   incremental: only papers not yet in the collection are embedded (`--rebuild` re-embeds all).
3. **generate**: for every paper and every column, retrieves the top-k chunks of *that
   paper only* using a column-specific query. All column contexts for one paper go into a
   single Groq call (`openai/gpt-oss-120b`) that returns JSON: `content`, `evidence_ids`,
   and a short verbatim `quote` per column.
4. **automatic check**: the cited chunk ids must exist and the quote must appear in a cited
   chunk (letters and digits only, so PDF line breaks do not matter; segments split on "..."
   are matched separately, with a fuzzy fallback). Labels: `quote_verified`, `quote_fuzzy`,
   `quote_not_found`, `no_evidence`, `not_stated`.
5. **review**: every cell is judged against the full text of its retrieved chunks and gets a
   verdict in `data/review/verdicts.csv`: `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`,
   `NOT_STATED_OK`, or `NOT_STATED_WRONG`. Cells that are not fully supported are replaced by
   a corrected text that uses only claims the chunks back. `python run_pipeline.py review`
   merges the verdicts and compares them with the automatic check.

`data/papers.csv` has two per-paper overrides. Some OpenAlex abstracts are wrong: they hold
only an author list, or another paper's abstract. For those, `use_openalex_abstract=no`
skips the abstract. `pdf_url` points to the ACL Anthology or arXiv PDF, or is `none` to skip
a PDF that OpenAlex lists but that is not the paper itself.

The **Paper** column (authors, year, title, venue, DOI) comes from `data/papers.csv`, which
was verified by hand against Crossref, OpenAlex, and publisher pages. The LLM never writes
it, so citations cannot be hallucinated.

Matrix columns: Paper, Problem, Domain & Data, RAG Method, Evaluation & Metrics,
Main Results, Limitations, Gap. Columns, retrieval queries, and instructions live in
`configs/default.yaml`.

## Results of this run

13 papers × 7 generated columns = 91 cells.

| Review verdict | Cells |
|---|---|
| SUPPORTED | 62 |
| PARTIAL | 23 |
| UNSUPPORTED | 2 |
| NOT_STATED_OK | 3 |
| NOT_STATED_WRONG | 1 |

26 cells (29%) needed correction. The automatic quote check is not enough on its own:
21 of the 72 cells labelled `quote_verified` were still partly or wholly wrong. The quote
was real, but the claim around it went beyond it. The most common errors were:

- weaknesses of prior work, or general RAG challenges, reported as the paper's own limitations;
- a dataset language ("English") stated when no chunk says so;
- a number attributed to the wrong method or condition.

The full cross-tabulation is in `results/review_vs_auto_check.csv`.

**How the review was done.** The verdicts were produced with AI assistance: separate
LLM reviewers that saw only the cell text and its retrieved chunks, not the
generator's reasoning. Not every verdict has been re-checked by a human. The review judges
cells against the retrieved chunks, not the full paper, so a cell can be correct for the
chunks and still miss something stated elsewhere in the paper.

## Setup

Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then put your Groq key in .env
```

Get a free API key at [console.groq.com](https://console.groq.com/). Never commit `.env`.

## Run

```bash
python run_pipeline.py fetch                      # abstracts + OA PDFs into data/cache/
python run_pipeline.py index                      # chunk, embed locally, store in Chroma
python run_pipeline.py generate --dry-run         # build prompts only, no API calls
python run_pipeline.py generate --only P01        # try one paper first
python run_pipeline.py generate                   # full matrix
python run_pipeline.py review                     # merge data/review/verdicts.csv
```

`python run_pipeline.py all` runs fetch, index, and generate.

### Staying within Groq free-tier limits

- One API call per paper, not one call per cell: 13 papers means 13 calls.
- Responses are cached in `results/llm_cache/` keyed by a hash of the model and prompt.
  A re-run with an unchanged config makes no API calls, and adding a paper only calls the
  API for that paper.
- `--dry-run` reports the approximate prompt size per paper before you spend tokens.
- Rate-limit errors are retried with backoff; `delay_seconds` spaces out calls.

## Reproducibility

- Seed `42` (config `seed`) is set for Python and NumPy, and passed to Groq. Temperature is `0`.
- `results/run_log.json` records the timestamp, seed, models, chunking, top-k, and the
  counts of each automatic check label.
- Hosted LLM output is not guaranteed to be bit-identical across runs even with a fixed
  seed. The cached responses in `results/llm_cache/` are the record of the run reported here.

## Repository layout

```
configs/default.yaml     all paths, hyperparameters, columns, prompts
data/papers.csv          verified paper list (input)
data/review/verdicts.csv cell-by-cell review verdicts and corrections
data/manual/             optional hand-added text per paper
src/gapmatrix/           fetch, index (chunk/embed/Chroma), generate (Groq + check), export/review
run_pipeline.py          command-line entry point
results/                 matrices, evidence, review summary, run log, LLM cache
                         (gap_matrix_ringkas.csv: condensed matrix used in the report)
```

## Limitations

- Three papers (P02, P06, P17) have no usable open-access PDF and
  rest on the abstract alone. Cells the abstract does not cover are marked `tidak disebut`
  (not stated) instead of being guessed.
- Fixed-size character chunking can split sentences. It is simple, and it is good enough
  for abstracts.
- The literature search covers OpenAlex, Crossref, and web search up to 2026-10-26. The
  paper list is limited to studies that test RAG for question answering over finance or
  banking documents, with English or Indonesian data. Paper IDs keep their original numbers,
  so the list has gaps (P03, P07, P10, P14, P18 to P22 were dropped).
