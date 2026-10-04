"""Intelligent Document Investigator - base version (ALG-AI-02)."""
import io, json, logging, os, re, uuid
from pathlib import Path

import fitz, httpx, numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

log = logging.getLogger("investigator")

try:
    from rank_bm25 import BM25Okapi
except Exception as e:  # optional: app still boots, retrieval falls back to token overlap
    BM25Okapi = None
    log.warning("rank_bm25 unavailable (%s); using token-overlap fallback.", e)

from conflicts import attach_hints, compare_claims, doc_subject_map, doc_subjects_for, extract_claims, filter_question_relevant, pair_candidates, verify_candidates

load_dotenv()
try:
    import pytesseract
    from PIL import Image
    import shutil
    OCR = shutil.which("tesseract") is not None  # python wrapper is useless without the binary
    if not OCR:
        log.warning("pytesseract is installed but the Tesseract binary was not found; image/scanned pages will report a warning.")
except Exception as e:
    OCR = False
    log.warning("OCR unavailable (%s); image/scanned pages will report a warning.", e)

def _public_doc(entry):
    """API-safe document record. Filesystem paths never leave the server."""
    return {k: entry[k] for k in ("id", "filename", "pages", "ocr_pages", "chunks", "warning") if k in entry}


BASE = Path(__file__).parent
UP = BASE / "uploads"; UP.mkdir(exist_ok=True)
DOCS, CHUNKS = {}, []
STATE = {"bm25": None, "emb": None}
_model = None
app = FastAPI(title="Intelligent Document Investigator")


# ---------- configuration ----------
def get_config():
    """Read runtime config. Never logs or returns secret values."""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini").strip().lower(),
        "llm_model": os.getenv("LLM_MODEL", ""),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "local").strip().lower(),
        "ocr_provider": os.getenv("OCR_PROVIDER", "local").strip().lower(),
        "min_sim": float(os.getenv("MIN_SIM", 0.40)),
    }


def check_config():
    """Fail fast with an actionable message for unsupported provider settings."""
    cfg = get_config()
    if cfg["llm_provider"] not in ("gemini", "openai", "groq"):
        raise HTTPException(500, f"Unsupported LLM_PROVIDER={cfg['llm_provider']!r}. Set LLM_PROVIDER to 'gemini', 'openai' or 'groq' in .env (see .env.example).")
    if cfg["embedding_provider"] not in ("local", "openai"):
        raise HTTPException(500, f"Unsupported EMBEDDING_PROVIDER={cfg['embedding_provider']!r}. Set EMBEDDING_PROVIDER to 'local' or 'openai' in .env (see .env.example).")
    if cfg["ocr_provider"] not in ("local",):
        raise HTTPException(500, f"Unsupported OCR_PROVIDER={cfg['ocr_provider']!r}. Phase 1 supports only 'local'.")
    return cfg


# ---------- embeddings (optional dense search; BM25 fallback always works) ----------
def embedder():
    global _model
    if _model is None:
        if get_config()["embedding_provider"] != "local":
            _model = False  # non-local provider: handled per-query by embed_openai()
        else:
            try:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            except Exception as e:
                log.warning("Local embedding model unavailable (%s); using BM25 only.", e)
                _model = False
    return _model


def embed_openai(texts):
    """OpenAI embeddings for EMBEDDING_PROVIDER=openai. Returns None (BM25 fallback) on any failure."""
    key = os.getenv("OPENAI_API_KEY") or (os.getenv("LLM_API_KEY") if get_config()["llm_provider"] == "openai" else None)
    if not key:
        log.warning("EMBEDDING_PROVIDER=openai but no OPENAI_API_KEY (or LLM_API_KEY) is set; using BM25 only.")
        return None
    try:
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        vecs = []
        for i in range(0, len(texts), 96):
            r = httpx.post("https://api.openai.com/v1/embeddings", headers={"Authorization": f"Bearer {key}"},
                           timeout=60, json={"model": model, "input": texts[i:i + 96]})
            r.raise_for_status()
            vecs.extend(d["embedding"] for d in r.json()["data"])
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        log.warning("OpenAI embeddings failed (%s); using BM25 only.", e)
        return None


def embed_texts(texts):
    """Encode texts per EMBEDDING_PROVIDER. Returns (matrix or None, kind). Never raises."""
    if not texts:
        return None, "none"
    cfg = get_config()
    if cfg["embedding_provider"] == "openai":
        m = embed_openai(texts)
        if m is not None:
            n = np.linalg.norm(m, axis=1, keepdims=True); n[n == 0] = 1
            return m / n, "openai"
        return None, "none"
    m = embedder()
    if not m:
        return None, "none"
    try:
        return np.array(m.encode(texts, normalize_embeddings=True)), "local"
    except Exception as e:
        log.warning("Local embedding encoding failed (%s); using BM25 only.", e)
        return None, "none"


def tok(s): return re.findall(r"[a-z0-9]+", s.lower())


# ---------- ingestion ----------
def needs_ocr(t):
    t = t.strip()
    return len(t) < 40 or len(t.split()) < 8 or sum(c.isalnum() for c in t) / max(len(t), 1) < 0.45


SUPPORTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt", ".md"}
# Canonical extraction methods. "ocr" must stay exact: the frontend keys its badge off it.
METHOD_NATIVE_PDF, METHOD_OCR, METHOD_TEXT = "native_pdf_text", "ocr", "text"


def ocr_image(img):
    if not OCR:
        return ""
    try:
        return pytesseract.image_to_string(img)
    except Exception as e:  # missing binary, corrupt image, etc.: never crash ingestion
        log.warning("OCR failed (%s); page text kept as-is.", e)
        return ""


def extract(path: Path, display=None):
    """Return [(page_no, text, method)] - native text first, OCR only if needed."""
    label = display or path.name
    ext = path.suffix.lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(path)
        except Exception as e:
            raise HTTPException(400, f"Could not read {label} as PDF ({e}). The file may be corrupt.")
        if len(doc) == 0:
            raise HTTPException(400, f"{label} is an empty PDF (0 pages). Nothing was indexed.")
        out = []
        for i, p in enumerate(doc, 1):
            try:
                t, m = p.get_text(), METHOD_NATIVE_PDF
            except Exception as e:
                log.warning("Native text extraction failed for %s p.%d (%s).", label, i, e)
                t, m = "", METHOD_NATIVE_PDF
            if needs_ocr(t) and OCR:
                try:
                    t = ocr_image(Image.open(io.BytesIO(p.get_pixmap(dpi=250).tobytes("png"))))
                    m = METHOD_OCR if t.strip() else METHOD_NATIVE_PDF
                except Exception as e:
                    log.warning("OCR render failed for %s p.%d (%s); native text kept.", label, i, e)
            out.append((i, t, m))
        return out
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        try:
            t = ocr_image(Image.open(path))
        except Exception as e:
            raise HTTPException(400, f"Could not read {label} as an image ({e}).")
        if not t.strip():
            if not OCR:
                raise HTTPException(400, f"{label} needs OCR but no OCR engine is available. "
                                         "Install Tesseract (https://tesseract-ocr.github.io) and the 'pytesseract' package.")
            raise HTTPException(400, f"{label} produced no readable text. The image may be blank or unreadable.")
        return [(1, t, METHOD_OCR)]
    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        raise HTTPException(400, f"Could not read {label} as text ({e}).")
    if not text.strip():
        raise HTTPException(400, f"{label} is empty. Nothing was indexed.")
    return [(i, t, METHOD_TEXT) for i, t in enumerate(text.split("\f"), 1)]


def chunk_page(doc_id, filename, page, text, method):
    out, section, buf = [], "", ""

    def flush():
        nonlocal buf
        if len(buf.strip()) > 20:
            cid = f"{doc_id}:p{page}:c{len(out)}"
            out.append({"cid": cid, "chunk_id": cid, "doc_id": doc_id, "filename": filename,
                        "page": page, "section": section, "method": method, "extraction_method": method,
                        "chunk_type": "paragraph", "ocr_confidence": None,
                        "text": (f"{section}\n" if section else "") + buf.strip()})
        buf = ""

    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para: continue
        if len(para) < 80 and "\n" not in para and not para.endswith("."):
            flush(); section = para; continue          # heading
        if len(buf) + len(para) > 1500: flush()
        buf += para + "\n"
    flush()
    return out


def rebuild_index():
    if BM25Okapi is not None:
        STATE["bm25"] = BM25Okapi([tok(c["text"]) for c in CHUNKS]) if CHUNKS else None
    else:
        STATE["bm25"] = None
    STATE["emb"], STATE["emb_kind"] = embed_texts([c["text"] for c in CHUNKS]) if CHUNKS else (None, "none")


def overlap_scores(q, docs):
    """Minimal token-overlap fallback when rank_bm25 is unavailable."""
    qt = set(tok(q))
    return np.array([float(len(qt & set(t))) for t in docs])


# ---------- retrieval: BM25 + dense, fused with RRF ----------
def retrieve(q, k=7):
    if not CHUNKS:
        return [], 0.0, None
    if STATE["bm25"] is not None:
        bm = STATE["bm25"].get_scores(tok(q))
    else:
        bm = overlap_scores(q, [tok(c["text"]) for c in CHUNKS])
    ranks = {}
    for i, r in enumerate(np.argsort(-bm)[:15]): ranks[int(r)] = ranks.get(int(r), 0) + 1 / (60 + i)
    sim_max = None
    if STATE.get("emb") is not None:
        try:
            qv, _ = embed_texts([q])
            if qv is not None:
                sims = STATE["emb"] @ qv[0]; sim_max = float(sims.max())
                for i, r in enumerate(np.argsort(-sims)[:15]): ranks[int(r)] = ranks.get(int(r), 0) + 1 / (60 + i)
        except Exception as e:
            log.warning("Dense retrieval failed (%s); BM25 scores only.", e)
    order = sorted(ranks, key=ranks.get, reverse=True)
    seen, top = set(), []
    for i in order:  # diversify + dedupe: one copy per identical text, spread across documents
        key = norm(CHUNKS[i]["text"])
        if key in seen:
            continue
        if len(top) >= k - 1 and CHUNKS[i]["doc_id"] in {CHUNKS[j]["doc_id"] for j in top}:
            continue
        seen.add(key); top.append(i)
        if len(top) >= k + 2: break
    ev = [dict(CHUNKS[i], sid=f"S{n}") for n, i in enumerate(top, 1)]
    return ev, float(bm.max(initial=0) if len(bm) else 0), sim_max


# ---------- LLM ----------
SYSTEM = """You are a document investigator. Answer ONLY from the evidence blocks; never use outside knowledge.
Return JSON with keys:
answer_status ("supported" | "conflicting" | "insufficient_evidence"),
answer (short, plain),
confidence_label ("high"|"medium"|"low"), confidence_reason,
citations (list of ids like "S1" that support the answer),
conflicts (list of {field, claim_a, citation_a, quote_a, claim_b, citation_b, quote_b, reason, likely_newer, newer_reason}),
missing_information (list of strings).
Rules:
- "conflicting" only when two sources give incompatible values for the SAME subject and field (different dates, amounts, statuses). Extra detail that does not contradict is not a conflict. Mind qualifiers such as draft, forecast, revised.
- For conflicts do NOT pick a winner. If a source explicitly says it revises/supersedes another or is dated later, set likely_newer to its id and explain in newer_reason; otherwise null.
- If the evidence does not establish the answer, use "insufficient_evidence" and say what is missing.
- Cite only provided ids. Quotes must be copied verbatim from the evidence.
- Do not use outside knowledge and do not invent facts. Every factual claim in the answer must be traceable to one of the cited ids above.
- Conflict discovery is handled by the backend: always return "conflicts": []. If the request lists VERIFIED CONFLICTS, set answer_status to "conflicting" and present them."""


def llm_json(user):
    return llm_with_system(SYSTEM, user)


def llm_with_system(system, user):
    cfg = check_config()
    prov, key = cfg["llm_provider"], os.getenv("LLM_API_KEY")
    if not key:
        raise HTTPException(500, "LLM_API_KEY is not set. Copy .env.example to .env and add a key.")
    model = os.getenv("LLM_MODEL", "") or ({"gemini": "gemini-3.8-flash", "groq": "llama-3.3-70b-versatile"}.get(prov, "gpt-5-mini"))
    try:
        if prov == "gemini":
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": key}, timeout=90,
                           json={"systemInstruction": {"parts": [{"text": system}]},
                                 "contents": [{"parts": [{"text": user}]}],
                                 "generationConfig": {"responseMimeType": "application/json", "temperature": 0}})
            r.raise_for_status()
            try:
                txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError) as e:
                log.warning("Unexpected Gemini response shape: %s", r.text[:300])
                raise HTTPException(502, f"The model ({model}) returned an unexpected response shape. Check the model name and try again.") from e
        else:
            url = "https://api.groq.com/openai/v1/chat/completions" if prov == "groq" else "https://api.openai.com/v1/chat/completions"
            r = httpx.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=90,
                           json={"model": model, "response_format": {"type": "json_object"},
                                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            r.raise_for_status()
            try:
                txt = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                log.warning("Unexpected OpenAI response shape: %s", r.text[:300])
                raise HTTPException(502, f"The model ({model}) returned an unexpected response shape. Check the model name and try again.") from e
    except HTTPException:
        raise
    except httpx.TimeoutException as e:
        raise HTTPException(502, f"The model ({model}) timed out. Try again in a moment.") from e
    except httpx.HTTPStatusError as e:
        status = e.response.status_code if e.response is not None else "?"
        log.warning("LLM API error %s for model %s: %s", status, model, (e.response.text[:300] if e.response is not None else e))
        if status in (401, 403):
            raise HTTPException(502, "The LLM API rejected the key (401/403). Check LLM_API_KEY and LLM_PROVIDER in .env.") from e
        if status == 404:
            raise HTTPException(502, f"The model ({model}) was not found. Check LLM_MODEL for provider '{prov}'.") from e
        raise HTTPException(502, f"The LLM API returned error {status}. Try again in a moment.") from e
    except httpx.HTTPError as e:
        raise HTTPException(502, "Could not reach the LLM API. Check your network and try again.") from e
    return parse_llm_json(txt)


def parse_llm_json(txt):
    """Parse model output; recover a `{...}` object if wrapped in fences/prose. Never raises."""
    clean = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip()).strip()
    try:
        out = json.loads(clean)
    except Exception:
        out = None  # fall through to brace recovery below
    else:
        if isinstance(out, dict):
            return out
        # Valid JSON but not an object (e.g. a list): do NOT carve a sub-object out of it.
        log.warning("Model returned non-object JSON: %s", (txt or "")[:200])
        raise HTTPException(502, "The model returned an unparseable response. Try asking again.")
    try:  # recovery: first balanced {...} span
        start = clean.find("{")
        if start >= 0:
            depth, instr, esc = 0, False, False
            for i in range(start, len(clean)):
                ch = clean[i]
                if instr:
                    if esc: esc = False
                    elif ch == "\\": esc = True
                    elif ch == '"': instr = False
                elif ch == '"': instr = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        out = json.loads(clean[start:i + 1])
                        if isinstance(out, dict):
                            return out
                        break
    except Exception:
        pass
    log.warning("Model returned invalid JSON: %s", (txt or "")[:300])
    raise HTTPException(502, "The model returned an unparseable response. Try asking again.")


def norm(s): return re.sub(r"\s+", " ", s or "").strip().lower()


def _doc_label(filename):
    """Short human label derived from a stored filename. Deterministic, display-only."""
    base = (filename or "document").rsplit(".", 1)[0]
    base = re.sub(r"[_-]+", " ", base).strip()
    base = re.sub(r"^\d+\s*", "", base).strip()
    return base or (filename or "document")


def build_summary(ev, claims, candidates, conflicts_out, cited_sids, by, status):
    """Deterministic pipeline counters. Every number comes from actual pipeline data;
    anything uncomputable is omitted, never invented."""
    claim_sids = {c.get("citation_id") for c in (claims or [])}
    compared = set()
    for p in (candidates or []):
        compared.add(p["claim_a"].get("claim_id"))
        compared.add(p["claim_b"].get("claim_id"))
    kind = STATE.get("emb_kind", "none")
    return {
        "documents_examined": len({e.get("doc_id") for e in (ev or [])}),
        "passages_retrieved": len(ev or []),
        "relevant_passages": len([e for e in (ev or []) if e.get("sid") in claim_sids]),
        "claims_extracted": len(claims or []),
        "claims_compared": len(compared),
        "conflicts_found": len(conflicts_out or []),
        "supporting_sources": len({by[s].get("doc_id") for s in (cited_sids or []) if s in (by or {})}),
        "retrieval": {"bm25": True,
                      "semantic": kind if kind in ("local", "openai") else "off",
                      "fusion": "RRF"},
        "status": status,
    }


def build_decision_reason(status, conflicts_out, cited_sids, by, missing, weak=False):
    """Why the final status was produced — deterministic facts only (claims,
    hints, citations, status). No LLM reasoning is used or exposed."""
    by = by or {}
    if status == "conflicting" and conflicts_out:
        fields = sorted({c.get("field") for c in conflicts_out})
        details = []
        for c in conflicts_out:
            a, b = c.get("a", {}), c.get("b", {})
            la = _doc_label(by.get(a.get("cite"), {}).get("filename"))
            lb = _doc_label(by.get(b.get("cite"), {}).get("filename"))
            va = re.sub(r"^Claim C\d+:\s*", "", a.get("claim", "?"))
            vb = re.sub(r"^Claim C\d+:\s*", "", b.get("claim", "?"))
            details.append("%s states %s; %s states %s." % (la, va, lb, vb))
        for c in conflicts_out:
            if c.get("likely_newer"):
                wlabel = _doc_label(by.get(c["likely_newer"], {}).get("filename"))
                nr = norm(c.get("newer_reason") or "")
                marker = ("supersedes the earlier document" if "supersede" in nr
                          else "describes itself as revised" if "revis" in nr
                          else "describes itself as updated")
                details.append("Likely-current hint: the %s source %s (hint only, not a verdict)."
                               % (wlabel, marker))
        details.append("The system does not automatically select one value.")
        return {"type": "conflict",
                "summary": "Documents state different values for: %s." % ", ".join(fields),
                "details": details}
    if status == "supported":
        cited = [s for s in (cited_sids or []) if s in by]
        docs = sorted({_doc_label(by[s].get("filename")) for s in cited})
        details = []
        for s in cited[:3]:
            e = by[s]
            details.append("%s (p. %s): %s" % (_doc_label(e.get("filename")), e.get("page"),
                                              (e.get("text") or "")[:140]))
        if len(cited) > 3:
            details.append("... and %d more citation(s)." % (len(cited) - 3))
        return {"type": "support",
                "summary": "Evidence from %d document(s) supports the answer with %d validated citation(s)."
                           % (len(docs), len(cited)),
                "details": details}
    details = []
    if weak:
        details.append("No passage matched the question closely enough.")
    else:
        details.append("Retrieved passages do not contain the requested fact.")
    for m in (missing or [])[:3]:
        details.append("Still needed: %s" % m)
    return {"type": "insufficient",
            "summary": "The uploaded documents do not establish the answer.",
            "details": details}


def verified_quote(q, text):
    """Show the model's quote only if it really appears in the source; otherwise fall back to source text."""
    return q.strip() if q and norm(q) in norm(text) else text[:300]


# ---------- Phase 4: investigation trail + evidence strength (deterministic) ----------
def _trail_claim(c):
    return {"claim_id": c["claim_id"], "citation_id": c["citation_id"], "filename": c["filename"],
            "page": c["page"], "subject": c["subject"], "attribute": c["attribute"],
            "value": c["value"], "value_norm": c["value_norm"], "value_type": c["value_type"],
            "qualifiers": c["qualifiers"]}


def _trail_docs(ev):
    seen, out = set(), []
    for e in ev:
        if e.get("doc_id") not in seen:
            seen.add(e["doc_id"])
            out.append({"doc_id": e["doc_id"], "filename": e["filename"]})
    return out


def build_trail(ev, claims, candidates, det_conflicts, cited, status, weak=False):
    """Structured, user-verifiable processing facts. No LLM reasoning inside."""
    docs = _trail_docs(ev)
    checks = []
    if weak or not ev:
        checks.append({"key": "evidence_retrieved", "state": "fail",
                       "label": "No passage matched the question closely enough."})
        checks.append({"key": "closest_preserved", "state": "pass" if ev else "warn",
                       "label": "Closest retrieved passages shown as evidence." if ev
                                else "No relevant passages retrieved."})
    else:
        checks.append({"key": "evidence_retrieved", "state": "pass",
                       "label": "%d relevant passage(s) retrieved from %d document(s)." % (len(ev), len(docs))})
    if status == "conflicting":
        multi = len({c["_a"]["doc_id"] for c in det_conflicts} |
                    {c["_b"]["doc_id"] for c in det_conflicts}) > 1 if det_conflicts else False
        checks.append({"key": "claims_multi_doc", "state": "pass" if multi else "warn",
                       "label": "Claims found in multiple documents." if multi
                                else "Conflicting claims come from a single document."})
        checks.append({"key": "same_subject", "state": "pass", "label": "Claims refer to the same subject."})
        checks.append({"key": "same_attribute", "state": "pass", "label": "Claims refer to the same attribute."})
        checks.append({"key": "normalized", "state": "pass", "label": "Values normalized successfully."})
        checks.append({"key": "values_differ", "state": "warn", "label": "Normalized values are incompatible."})
        inconclusive = any("inconclusive" in (c.get("reason") or "") for c in det_conflicts)
        checks.append({"key": "verified", "state": "warn" if inconclusive else "pass",
                       "label": "Verifier inconclusive; shown side-by-side unresolved." if inconclusive
                                else "Conflict verified against source context."})
        if any(c.get("likely_newer") for c in det_conflicts):
            checks.append({"key": "hint", "state": "info",
                           "label": "A likely-current hint was found (explicit revised/supersedes wording)."})
    elif status == "supported":
        checks.append({"key": "supporting_claim", "state": "pass", "label": "Direct supporting claim found."})
        checks.append({"key": "citations_validated", "state": "pass" if cited else "fail",
                       "label": "%d citation(s) validated against stored evidence." % len(cited) if cited
                                else "No valid citation for the answer."})
        checks.append({"key": "no_conflict", "state": "pass", "label": "No conflicting claim detected."})
    else:
        checks.append({"key": "completed", "state": "pass", "label": "Investigation completed."})
        checks.append({"key": "no_direct_evidence", "state": "warn",
                       "label": "No evidence directly establishes the requested claim."})
        checks.append({"key": "closest_preserved", "state": "pass" if cited else "warn",
                       "label": "Closest relevant evidence preserved." if cited
                                else "No relevant passages to show."})
    return {"retrieved_evidence_count": len(ev), "documents_considered": docs,
            "claims": [_trail_claim(c) for c in claims],
            "candidates_examined": len(candidates), "verified_conflicts": len(det_conflicts),
            "weak_evidence": weak, "checks": checks}


def classify_strength(status, conflicts, cited_sids, by):
    """Deterministic evidence strength. Never assigned by the LLM, never a percentage."""
    docs = {by[s].get("doc_id") for s in (cited_sids or []) if s in by}
    n, nd = len(cited_sids or []), len(docs)
    if conflicts:
        fields = sorted({c.get("field") for c in conflicts})
        return {"level": "CONFLICTING",
                "reasons": ["Evidence contains incompatible claims about %s." % ", ".join(fields)]}
    if status == "insufficient_evidence":
        return {"level": "INSUFFICIENT",
                "reasons": ["No retrieved evidence directly establishes the answer."]}
    if n >= 2 and nd >= 2:
        return {"level": "STRONG",
                "reasons": ["%d supporting passages from %d documents; all citations validated." % (n, nd)]}
    if n >= 2:
        return {"level": "STRONG",
                "reasons": ["%d supporting passages; all citations validated." % n]}
    if n == 1:
        return {"level": "MODERATE",
                "reasons": ["A single supporting passage; no contradiction found."]}
    return {"level": "INSUFFICIENT", "reasons": ["No valid citation supports the answer."]}


# ---------- API ----------
@app.post("/documents")
async def upload(files: list[UploadFile] = File(...)):
    if not files or all(not (f.filename or "").strip() for f in files):
        raise HTTPException(400, "No files received. Choose at least one PDF, image, or text file.")
    check_config()
    saved = []
    for f in files:
        name = (f.filename or "unnamed").strip()
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(400, f"{name}: unsupported file type {ext or '(none)'!r}. "
                                     f"Supported: PDF, PNG, JPG, JPEG, WEBP, TXT, MD.")
        doc_id = uuid.uuid4().hex[:6]
        path = UP / f"{doc_id}_{Path(name).name}"
        try:
            path.write_bytes(await f.read())
        except Exception as e:
            raise HTTPException(400, f"{name}: could not save the upload ({e}).")
        try:
            pages = extract(path, display=name)
        except HTTPException as e:  # one bad file must not corrupt the index
            try: path.unlink(missing_ok=True)
            except Exception: pass
            raise  # message already names the file (extract uses display=name)
        except Exception as e:
            try: path.unlink(missing_ok=True)
            except Exception: pass
            log.warning("Ingestion failed for %s: %s", name, e)
            raise HTTPException(400, f"{name}: could not be read ({e}). The file may be corrupt.")
        new_chunks = []
        for n, t, m in pages:
            if t and t.strip():
                new_chunks.extend(chunk_page(doc_id, name, n, t, m))
        entry = {"id": doc_id, "filename": name, "path": str(path), "pages": len(pages),
                 "ocr_pages": sum(1 for p in pages if p[2] == METHOD_OCR),
                 "chunks": len(new_chunks),
                 "warning": None if new_chunks else "No indexable text found (blank, unreadable, or OCR unavailable)."}
        if not new_chunks:
            log.warning("Document %s yielded no chunks.", name)
        DOCS[doc_id] = entry
        CHUNKS.extend(new_chunks)
        saved.append(_public_doc(entry))
    rebuild_index()
    return saved


@app.get("/documents")
def docs(): return [_public_doc(d) for d in DOCS.values()]


@app.get("/documents/{doc_id}/pages/{page}")
def page(doc_id: str, page: int):
    d = DOCS.get(doc_id)
    if not d: raise HTTPException(404, "Document not found. It may have been removed when the server restarted (storage is in-memory).")
    if page < 1: raise HTTPException(404, f"Page {page} does not exist.")
    p = Path(d["path"])
    if not p.exists(): raise HTTPException(404, "Source file is missing from the server.")
    ext = p.suffix.lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(p)
        except Exception:
            raise HTTPException(404, "Source PDF can no longer be opened.")
        if page > len(doc): raise HTTPException(404, f"Page {page} does not exist (document has {len(doc)} page(s)).")
        return Response(doc[page - 1].get_pixmap(dpi=110).tobytes("png"), media_type="image/png")
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        if page != 1: raise HTTPException(404, f"Page {page} does not exist (images have a single page).")
        return FileResponse(p)
    try:
        parts = p.read_text(errors="ignore").split("\f")
    except Exception:
        raise HTTPException(404, "Source text can no longer be read.")
    if page > len(parts): raise HTTPException(404, f"Page {page} does not exist (document has {len(parts)} page(s)).")
    return PlainTextResponse(parts[page - 1])


class Q(BaseModel):
    question: str


class CompareReq(BaseModel):
    doc_ids: list


@app.post("/compare")
def compare(req: CompareReq):
    """Deterministic document comparison. Reuses the claim engine (no second
    implementation, no LLM): same extraction, normalization, pairing and hint
    rules as investigation. Fully deterministic, no API key needed."""
    ids = req.doc_ids
    if not isinstance(ids, list) or not ids:
        raise HTTPException(400, "Select at least two documents to compare.")
    uniq = []
    for d in ids:
        if not isinstance(d, str) or not d.strip():
            raise HTTPException(400, "Document IDs must be non-empty strings.")
        if d not in uniq:
            uniq.append(d)
    if len(uniq) < 2:
        raise HTTPException(400, "Select at least two different documents to compare.")
    if len(uniq) > 8:
        raise HTTPException(400, "Comparison supports at most 8 documents at a time.")
    missing = [d for d in uniq if d not in DOCS]
    if missing:
        raise HTTPException(404, f"Document not found: {missing[0]}. It may have been deleted or the server restarted (storage is in-memory).")
    want = set(uniq)
    pseudo = []
    for ch in CHUNKS:
        if ch.get("doc_id") in want:
            e = dict(ch)
            e["sid"] = ch["cid"]  # backend-controlled id; pairing needs distinct ids per chunk
            pseudo.append(e)
    docs = [{"doc_id": d, "filename": DOCS[d]["filename"], "pages": DOCS[d]["pages"]} for d in uniq]
    if not pseudo:
        return {"documents": docs, "rows": [], "summary": {"compatible": 0, "conflicts": 0, "needs_review": 0, "info": 0},
                "note": "Selected documents contain no indexable text to compare."}
    try:
        claims = extract_claims(pseudo, doc_subject_map([c for c in CHUNKS if c.get("doc_id") in want]))
        by_cid = {e["sid"]: e for e in pseudo}
        doc_texts = {}
        for ch in CHUNKS:
            if ch.get("doc_id") in want:
                doc_texts.setdefault(ch["doc_id"], []).append(ch.get("text", ""))
        rows = compare_claims(claims, by_cid, {d: "\n".join(t) for d, t in doc_texts.items()})
    except Exception as e:
        log.warning("Comparison failed (%s).", e)
        raise HTTPException(500, "Comparison failed for these documents. Try again.")
    summary = {"compatible": 0, "conflicts": 0, "needs_review": 0, "info": 0}
    for r in rows:
        summary[{"compatible": "compatible", "conflict": "conflicts",
                 "needs_review": "needs_review", "info": "info"}[r["verdict"]]] += 1
    return {"documents": docs, "rows": rows, "summary": summary,
            "note": "Deterministic comparison only (no LLM verdicts). 'Needs review' rows have contrasting qualifiers."}


@app.post("/investigate")
def investigate(q: Q):
    question = (q.question or "").strip()
    if not question:
        raise HTTPException(400, "Question is empty. Type a question about your documents first.")
    if not CHUNKS: raise HTTPException(400, "Upload documents first.")
    cfg = check_config()
    ev, bm_max, sim_max = retrieve(question)
    if not ev:
        return {"status": "insufficient_evidence", "answer": "Not found in the uploaded documents.",
                "confidence": "high", "confidence_reason": "No passage matched the question.",
                "citations": [], "conflicts": [], "missing": [],
                "investigation": build_trail([], [], [], [], [], "insufficient_evidence", weak=True),
                "evidence_strength": classify_strength("insufficient_evidence", [], [], {}),
                "investigation_summary": build_summary([], [], [], [], [], {}, "insufficient_evidence"),
                "decision_reason": build_decision_reason("insufficient_evidence", [], [], {}, [], weak=True)}
    meta = lambda e: {"id": e["sid"], "doc_id": e["doc_id"], "filename": e["filename"], "page": e["page"],
                      "section": e["section"], "method": e["method"], "text": e["text"][:500]}
    weak = bm_max <= 0 and (sim_max is None or sim_max < cfg["min_sim"])
    if weak:
        by0 = {e["sid"]: e for e in ev}
        return {"status": "insufficient_evidence", "answer": "Not found in the uploaded documents.",
                "confidence": "high", "confidence_reason": "No passage matched the question.",
                "citations": [meta(e) for e in ev[:3]], "conflicts": [], "missing": [],
                "investigation": build_trail(ev, [], [], [], [e["sid"] for e in ev[:3]],
                                             "insufficient_evidence", weak=True),
                "evidence_strength": classify_strength("insufficient_evidence", [], [], {}),
                "investigation_summary": build_summary(ev, [], [], [], [e["sid"] for e in ev[:3]],
                                                       by0, "insufficient_evidence"),
                "decision_reason": build_decision_reason("insufficient_evidence", [], [e["sid"] for e in ev[:3]],
                                                         by0, [], weak=True)}

    block = "\n\n".join(f"[{e['sid']}] {e['filename']}, page {e['page']}, section: {e['section'] or '-'}"
                        f" ({e['method']})\n{e['text']}" for e in ev)
    by = {e["sid"]: e for e in ev}

    # ---------- Phase 2: deterministic-first conflict pipeline ----------
    # Python finds candidates (extraction + normalization + pairing); the LLM
    # only verifies them. The model never discovers conflicts on its own.
    det_conflicts = []
    claims_all = []
    try:
        evidence_doc_ids = {e.get("doc_id") for e in ev}
        doc_subs = {d: s for d, s in doc_subject_map(CHUNKS).items() if d in evidence_doc_ids}
        claims_all = extract_claims(ev, doc_subs)
        candidates = pair_candidates(claims_all)
        candidates = filter_question_relevant(candidates, question)
    except Exception as e:
        log.warning("Claim extraction failed (%s); continuing without deterministic conflicts.", e)
        candidates = []
    if candidates:
        det_conflicts = verify_candidates(candidates, llm_with_system)
        doc_texts = {}
        for ch in CHUNKS:
            if ch.get("doc_id") in evidence_doc_ids:
                doc_texts.setdefault(ch["doc_id"], []).append(ch.get("text", ""))
        attach_hints(det_conflicts, by, {d: "\n".join(t) for d, t in doc_texts.items()})

    if det_conflicts:
        det_section = "\n".join(
            "- CONFLICT on '%s' [%s]: %r (%s) vs [%s]: %r (%s).%s" % (
                c["field"], c["citation_a"], c["_a"]["source_text"], c["claim_a"],
                c["citation_b"], c["_b"]["source_text"], c["claim_b"],
                (" Hint: %s" % c["newer_reason"]) if c["newer_reason"] else "")
            for c in det_conflicts)
        answer_user = (f"Question: {q.question}\n\nEvidence:\n{block}\n\n"
                       f"VERIFIED CONFLICTS (found deterministically, already checked):\n{det_section}\n\n"
                       "You MUST set answer_status to \"conflicting\", describe each conflicting claim "
                       "with its citation id, and return \"conflicts\": [] (the backend attaches the "
                       "verified conflicts). Do not pick a winner and do not invent other conflicts.")
    else:
        answer_user = (f"Question: {q.question}\n\nEvidence:\n{block}\n\n"
                       "The checker found no conflicting claims. Answer from the evidence "
                       "(supported or insufficient_evidence) and return \"conflicts\": [].")
    raw = llm_json(answer_user)

    conflicts = []
    for c in det_conflicts:  # citation validator: only backend-supplied ids reach the UI
        a, b = c.get("citation_a"), c.get("citation_b")
        if a in by and b in by and a != b:
            nw = c.get("likely_newer") if c.get("likely_newer") in by else None
            conflicts.append({"field": c.get("field"), "reason": c.get("reason"),
                              "conflict_type": c.get("conflict_type"), "resolution": "unresolved",
                              "likely_newer": nw, "newer_reason": c.get("newer_reason") if nw else None,
                              "a": {"claim": "Claim %s: %s" % (c["_a"]["claim_id"], c.get("claim_a")),
                                    "cite": a, "quote": verified_quote(c["_a"]["source_text"], by[a]["text"])},
                              "b": {"claim": "Claim %s: %s" % (c["_b"]["claim_id"], c.get("claim_b")),
                                    "cite": b, "quote": verified_quote(c["_b"]["source_text"], by[b]["text"])}})
    cited = [s for s in (raw.get("citations") or []) if s in by]
    for c in conflicts: cited += [c["a"]["cite"], c["b"]["cite"]]
    cited = list(dict.fromkeys(cited))
    status = raw.get("answer_status")
    if conflicts:
        status = "conflicting"  # verified conflicts cannot be overridden by the model
    elif status == "conflicting":
        status = "supported" if cited else "insufficient_evidence"
    if status == "supported" and not cited: status = "insufficient_evidence"
    if status not in ("supported", "conflicting", "insufficient_evidence"): status = "insufficient_evidence"
    return {"status": status, "answer": raw.get("answer", ""), "confidence": raw.get("confidence_label", "low"),
            "confidence_reason": raw.get("confidence_reason", ""),
            "citations": [meta(by[s]) for s in (cited or list(by)[:3])], "conflicts": conflicts,
            "missing": raw.get("missing_information") or [],
            "investigation": build_trail(ev, claims_all, candidates, det_conflicts, cited, status),
            "evidence_strength": classify_strength(status, conflicts, cited, by),
            "investigation_summary": build_summary(ev, claims_all, candidates, conflicts, cited, by, status),
            "decision_reason": build_decision_reason(status, conflicts, cited, by,
                                                     raw.get("missing_information") or [])}


@app.get("/test-report")
def test_report():
    """Return the latest evaluation report. Read-only: never runs the suite."""
    p = BASE / "tests" / "report.json"
    if not p.exists():
        raise HTTPException(404, "No evaluation report yet. Run 'python -m tests.run_eval' first.")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning("Could not read test report (%s).", e)
        raise HTTPException(500, "Stored evaluation report is unreadable. Re-run 'python -m tests.run_eval'.")


@app.get("/")
def index():
    static = BASE / "static" / "index.html"  # canonical location
    legacy = BASE / "index.html"              # current repo layout
    if static.exists(): return FileResponse(static)
    if legacy.exists(): return FileResponse(legacy)
    raise HTTPException(500, "Frontend file index.html is missing on the server.")
