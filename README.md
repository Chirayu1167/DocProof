# Intelligent Document Investigator (ALG-AI-02)

Citation-first, conflict-aware RAG over PDF, scanned PDF/images, and text documents.
Three answer states: `supported`, `conflicting`, `insufficient_evidence`.

## First run

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # then put your key in LLM_API_KEY
uvicorn main:app --reload
```

Open http://127.0.0.1:8000, upload the demo files
(`1_vendor_proposal.txt`, `2_steering_minutes.txt`, `3_revised_schedule.txt`), and ask:

- "What is the production release date?" → `conflicting` (30 June 2027 vs 15 July 2027)
- "What is the budget?" → `conflicting` (INR 12,00,000 vs INR 14,50,000)
- "Who owns the trademark?" → `insufficient_evidence`

No Redis, Postgres, Docker, or external vector DB needed. Single process, in-memory index.

## What actually exists (Phase 1)

- `main.py`: FastAPI backend — upload, PyMuPDF extraction with OCR fallback,
  paragraph chunking, BM25 + optional local BGE embeddings fused with RRF,
  grounded LLM answers, citation-ID validation, per-page source preview.
- `index.html`: single-page UI served at `/`.
- Retrieval works BM25-only when `sentence-transformers` (or the model download)
  is unavailable; OCR pages fall back to native text when Tesseract is missing
  (image-only docs then report a warning instead of indexing nothing silently).
- `conflicts.py`: deterministic-first conflict pipeline — claim extraction from
  retrieved evidence only; date/amount/percentage/version/quantity normalization
  (ambiguous inputs left as-is, never guessed); per-row table extraction so table
  cells can't steal claims; attribute keywords matched whole-word with document-
  title references ("Production Release Memo") excluded; candidate pairing on same
  subject + attribute + value type with qualifier/period/currency guards;
  subject mentions in headings/titles are weak and never veto cross-document
  pairing (two differing explicitly-asserted subjects still never pair);
  revision-framed sentences ("revised from X to Y") are tagged historical and never
  form same-document pairs; question-relevance filtering; LLM verifies candidates
  only (`conflict`/`compatible`/`insufficient_context`); verifier failures fail open
  to `unresolved` side-by-side display; `likely_newer` hints only from explicit
  signals (supersedes/revised/updated wording), never verdicts.
- `tests/test_conflicts.py`: 25 unit tests for the above (`python -m pytest tests/ -q`).
- `tests/tests.json`: 12-case evaluation manifest (conflict, qualifier, hint,
  supported, cross-document, table, compatible, unanswerable, ambiguous, OCR,
  citation cases) with filename/page/snippet ground truth.
- `tests/run_eval.py`: runs the manifest through the real pipeline
  (upload + retrieve + deterministic conflicts + citation validation) and writes
  `tests/report.json`. Stubbed by default (`python -m tests.run_eval` — no key,
  no cost); live model via `EVAL_MODE=live` (requires `LLM_API_KEY`). The report
  always labels its mode; stubbed results are never presented as model results.
- `GET /test-report`: returns the latest report (404 with instructions if none);
  the UI has an Evaluation section that renders pass/fail, metrics and failures.
- Storage is in-memory: restarting the server clears documents. Uploaded bytes
  are kept under `uploads/`.

## Configuration (`.env`)

| Key | Values | Notes |
|---|---|---|
| `LLM_PROVIDER` | `gemini` \| `openai` | Anything else → clear 500 error, no crash |
| `LLM_API_KEY` | secret | Never commit; missing key → actionable error on query |
| `LLM_MODEL` | model name | Empty = provider default |
| `EMBEDDING_PROVIDER` | `local` \| `openai` | Failures fall back to BM25 |
| `OCR_PROVIDER` | `local` | Requires Tesseract binary + `pytesseract` |
| `MIN_SIM` | float | Dense-similarity floor for the no-match short-circuit |

Restart the server after changing providers.

## Known limits

- Deterministic extraction covers explicit date/amount/percentage/version/quantity
  statements with recognizable attribute keywords; purely textual disagreements
  (e.g. status wording) are not paired.
- Verifier and answer wording quality depend on the configured LLM.
- Chunking has no overlap; very long documents may exceed the LLM context.
- Quote fallback shows the first 300 chars of the source chunk when the model's
  quote doesn't match stored text.

## Evaluation (Phase 3)

```powershell
python -m pytest tests/ -q   # unit tests: normalization, pairing, verifier, hints
python -m tests.run_eval     # stubbed end-to-end eval -> tests/report.json
EVAL_MODE=live python -m tests.run_eval  # real LLM (needs LLM_API_KEY)
```

What is measured (see `tests/report.json` → `metrics`):

| Metric | Meaning |
|---|---|
| `citation_validity` | returned citations that resolve to real docs/pages/chunks with genuine quotes |
| `citation_recall` / `recall_at_5` / `recall_at_10` | expected evidence snippets found in cited / top-5 / top-10 retrieved passages |
| `conflict_precision` / `recall` / `F1` | conflict vs compatible calls on single-status ground truth (ambiguous excluded) |
| `abstention_precision` / `recall` | correct `insufficient_evidence` behaviour, plus false-confident count |
| `unsupported_claim_rate` | answered cases with zero valid citations (0 is good) |

Latency (mean/median; p95 only with ≥20 samples) is reported, never pass/fail.
The OCR case is skipped with reason when no Tesseract engine is installed.

Latest stubbed run: 11/11 passed,
1 skipped (`ocr_site_review`, no Tesseract binary); citation validity 100%,
recall@5/@10 100%, conflict P/R/F1 1.0, abstention P/R 1.0, unsupported-claim
rate 0. Retrieval, extraction, pairing, hints, validation and status
enforcement are real in this mode; only the LLM wording/verdicts are stubbed
(the configured provider key is rejected by the provider, see below).

Unit suite: 57 tests pass (`python -m pytest tests/ -q`), covering normalization,
pairing, verifier behavior, hints, comparison, investigation trail/strength,
`investigation_summary`, `decision_reason`, claim/citation provenance, API safety,
and cross-document revision-narrative regression (`tests/test_crossdoc_regression.py`
with generic `tests/fixtures/harbor_*.txt` fixtures: conflicting dates/budget with
supersedes hint, compatible pilot, unanswerable question).

## Validation status (final phase)

- Live LLM: validated end-to-end with the real provider (Gemini) on uploaded PDFs:
  the production-release question returned `conflicting` (15 April 2026 vs
  30 April 2026) with correct per-document provenance and the supersedes hint kept.
  The free-tier key is occasionally rate-limited (HTTP 429); the app answers those
  with a clean controlled 502 — no crash, no key material, no stack trace.
  All deterministic stages (retrieval, claims, pairing, hints, validation, status
  enforcement) run for real in both stubbed and live modes.
- OCR: no Tesseract binary in this environment. Uploading the scanned fixture
  (`tests/fixtures/scan_memo.png`) returns HTTP 400 with an actionable message
  naming the missing OCR engine (covered by automated test). Native PDF/text
  extraction is unaffected.
- Demo scenarios (stubbed LLM wording, real engine): release-date question →
  `conflicting` (30 June 2027 vs 15 July 2027, supersedes hint kept); pilot
  question → `supported`; trademark question → `insufficient_evidence` with no
  manufactured conflict. Live `/compare` on the demo corpus returns 2 conflicts
  (budget, release date, both hinted to the revised schedule) + 1 compatible row
  (pilot date).

## Run it

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # then put your key in LLM_API_KEY
uvicorn main:app --reload     # serves API + UI at http://127.0.0.1:8000
python -m pytest tests/ -q    # 48 unit/integration tests, no key needed
python -m tests.run_eval      # stubbed end-to-end eval -> tests/report.json
```

No Redis, Postgres, Docker, FAISS, vector DB, NLI model, agents, auth, or database.
Single process, in-memory index; restart clears documents (files remain under `uploads/`).

## Investigation experience (Phase 4)

- `POST /compare` (`{"doc_ids": [...]}`): deterministic document comparison reusing
  the claim engine — no second implementation, no LLM, no API key needed. Rows are
  `compatible` / `conflict` / `needs_review` (contrasting qualifiers) / `info`
  (different contexts, e.g. periods/currencies); supersedes hints preserved.
  Rejects empty/single/duplicate/unknown selections with clear errors.
- Every `/investigate` answer carries `investigation` (evidence count, documents,
  claims with normalized values, candidates examined, deterministic `checks`) and
  `evidence_strength` (`STRONG` / `MODERATE` / `INSUFFICIENT` / `CONFLICTING`
  with reasons — deterministic, never a percentage, never LLM-assigned).
- `investigation_summary`: deterministic pipeline counters (`documents_examined`,
  `passages_retrieved`, `relevant_passages`, `claims_extracted`, `claims_compared`,
  `conflicts_found`, `supporting_sources`, `retrieval` method flags, `status`).
- `decision_reason`: deterministic `{type, summary, details}` explaining the final
  status from claims, hints and citations only — no LLM reasoning, no invented text.
- The UI adds: document checkboxes + Compare table (✓/⚠/ℹ with clickable
  provenance), staged processing status while investigating, an Investigation flow
  section (retrieved → claims → compared → result, with the real retrieval methods),
  CLAIM A / CLAIM B cards with an explicit `≠` conflict line, supporting-claim cards,
  `↻ REVISED SOURCE` tags on hinted sources, an expandable "Why this result?"
  section (decision facts + checks), stored-passage citations with doc/page preview
  headers, and an examined-evidence line on insufficient results.
- API responses never include server filesystem paths (`/documents`, upload,
  `/compare`, citations carry ids/filenames/pages only).

## Future work (NOT implemented — do not assume otherwise)

- Phase 2 remainder: NLI-model verification, version-aware conflict engine.
- Larger live-model eval, FAISS, cross-encoder reranking, advanced chunk overlap, deployment.
