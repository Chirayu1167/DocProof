"""Deterministic-first conflict detection (Phase 2, ALG-AI-02).

Pipeline: extract claims from retrieved evidence -> normalize values ->
pair candidates deterministically -> LLM verifies each candidate only.

The LLM never discovers conflicts; it only classifies supplied pairs as
"conflict" | "compatible" | "insufficient_context".
"""
import logging
import re

log = logging.getLogger("investigator")

# ---------------------------------------------------------------- normalization helpers

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"])}
for _a, _i in list(MONTHS.items()):
    MONTHS[_a[:3]] = _i  # jan, feb, ...
MONTHS["sept"] = 9


def _norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def normalize_date(s):
    """Deterministic date -> 'YYYY-MM-DD'. Returns None when unsafe/ambiguous."""
    t = (s or "").strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", t)
    if m:
        y, mo, d = map(int, m.groups())
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None
    m = re.match(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)\s*,?\s*(\d{4})$", t)
    if m:
        d, mon, y = m.groups()
        mo = MONTHS.get(mon.lower())  # full names and 3-letter abbrs both mapped
        if mo and 1 <= int(d) <= 31:
            return f"{int(y):04d}-{mo:02d}-{int(d):02d}"
        return None
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s*(\d{4})$", t)
    if m:
        mon, d, y = m.groups()
        mo = MONTHS.get(mon.lower())
        if mo and 1 <= int(d) <= 31:
            return f"{int(y):04d}-{mo:02d}-{int(d):02d}"
        return None
    m = re.match(r"^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})$", t)
    if m:
        a, b, y = map(int, m.groups())
        if a > 12 and 1 <= b <= 12 and 1 <= a <= 31:
            d, mo = a, b          # D/M/Y, unambiguous
        elif b > 12 and 1 <= a <= 12 and 1 <= b <= 31:
            d, mo = b, a          # M/D/Y, unambiguous
        else:
            return None           # both <= 12: ambiguous, refuse to guess
        return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


CURRENCIES = {"inr": "INR", "₹": "INR", "rs": "INR", "rs.": "INR", "rupee": "INR", "rupees": "INR",
              "usd": "USD", "$": "USD", "dollar": "USD", "dollars": "USD",
              "eur": "EUR", "€": "EUR", "euro": "EUR", "euros": "EUR",
              "gbp": "GBP", "£": "GBP", "pound": "GBP", "pounds": "GBP"}
MULTIPLIERS = {"thousand": 1e3, "k": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
               "million": 1e6, "millions": 1e6, "m": 1e6, "mn": 1e6,
               "crore": 1e7, "crores": 1e7, "cr": 1e7,
               "billion": 1e9, "billions": 1e9, "bn": 1e9}


def normalize_amount(s):
    """Deterministic amount -> (CURRENCY, number). No FX conversion. None if unclear."""
    t = (s or "").strip()
    m = re.match(r"^([A-Za-z₹$€£.]+)?\s*([\d,]+(?:\.\d+)?)\s*([A-Za-z.]+)?$", t)
    if not m:
        return None
    cur_raw, num_raw, unit_raw = m.groups()
    cur = CURRENCIES.get((cur_raw or "").strip().lower().rstrip("."), None) if cur_raw else None
    if cur_raw and cur is None:
        # Unknown prefix: only tolerable when it is actually part of nothing meaningful.
        return None
    mult = 1.0
    if unit_raw:
        mult = MULTIPLIERS.get(unit_raw.strip().lower().rstrip("."), None)
        if mult is None:
            return None  # unknown unit: do not guess
    try:
        return cur, round(float(num_raw.replace(",", "")) * mult, 2)
    except ValueError:
        return None


def normalize_percentage(s):
    """'10%' / '10 percent' -> 10.0. None otherwise."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(%|percent)\s*$", (s or ""), re.IGNORECASE)
    return float(m.group(1)) if m else None


def normalize_version(s):
    """'v2' / 'Version 2' / 'v2.0' -> (2,) canonical (trailing zeros stripped)."""
    m = re.match(r"^\s*(?:v|version)\s*(\d+(?:\.\d+)*)\s*$", (s or ""), re.IGNORECASE)
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def normalize_quantity(s):
    """'100 units' -> (100.0, 'units'). Bare numbers are NOT quantities (too noisy)."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\s*$", (s or ""))
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


# ---------------------------------------------------------------- extraction patterns

_DATE_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_PATTERNS = {
    "date": [
        re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_DATE_MONTH}\s*,?\s*\d{{4}}\b", re.IGNORECASE),
        re.compile(rf"\b{_DATE_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s*\d{{4}}\b", re.IGNORECASE),
        re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
        re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{4}\b"),
    ],
    "amount": [
        re.compile(r"(?:INR|₹|Rs\.?|USD|\$|EUR|€|GBP|£)\s*[\d,]+(?:\.\d+)?\s*(?:lakh?s?|lac?s?|crore?s?|cr\.?|thousand|million|millions|billion|billions|[kmb])?\b", re.IGNORECASE),
        re.compile(r"\b[\d,]+(?:\.\d+)?\s*(?:lakh?s?|lac?s?|crore?s?|cr\.?)\b", re.IGNORECASE),
    ],
    "percentage": [re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent)\b", re.IGNORECASE)],
    "version": [re.compile(r"\b(?:v|version)\s*\d+(?:\.\d+)*\b", re.IGNORECASE)],
    "quantity": [re.compile(r"\b\d+(?:\.\d+)?\s*(?:units?|items?|pieces?|users?|seats?|licenses?|licences?|headcount|capacity)\b", re.IGNORECASE)],
}

DATE_KEYS = ["production release", "release date", "launch date", "pilot date", "pilot",
             "go live", "go-live", "delivery date", "delivery timeline",
             "deadline", "due date", "timeline", "schedule"]
AMOUNT_KEYS = ["budget", "cost", "price", "fee", "amount", "payment", "revenue", "expense"]
VERSION_KEYS = ["version"]
PERCENT_KEYS = ["percent", "percentage", "rate", "margin", "share", "growth"]
QUANTITY_KEYS = ["quantity", "units", "headcount", "capacity", "users"]
TYPE_KEYS = {"date": DATE_KEYS, "amount": AMOUNT_KEYS, "version": VERSION_KEYS,
             "percentage": PERCENT_KEYS, "quantity": QUANTITY_KEYS}

QUALIFIERS = {"draft", "revised", "approved", "proposed", "forecast", "actual",
              "previous", "current", "original", "final", "initial", "updated"}
# Release-family aliases collapse to one canonical attribute.
_ATTR_ALIASES = {"production release": "release date", "launch date": "release date",
                 "go live": "release date", "go-live": "release date",
                 "delivery date": "release date", "pilot": "pilot date"}
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_SUBJECT_RE = re.compile(r"[Pp]roject\s+([A-Z][\w\-]*)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'])")
_STOPWORDS = {"what", "is", "the", "a", "an", "of", "for", "about", "tell", "me", "please",
             "show", "who", "whom", "whose", "owns", "own", "when", "where", "which", "how",
             "much", "many", "does", "do", "did", "are", "was", "were", "be", "been", "give",
             "list", "find", "and", "or", "in", "on", "at", "to", "by", "from", "there", "here",
             "it", "its", "this", "that", "s"}
# Generic type words must NOT establish relevance on their own: "date" would
# otherwise link a pilot-date question to a production-release conflict.
_GENERIC_TOKENS = {"date", "dates", "amount", "amounts", "value", "values", "time",
                   "number", "numbers", "day", "days", "year", "years", "detail",
                   "details", "info", "information"}

MAX_CLAIMS_PER_EVIDENCE = 6


def _norm_key(s):
    s = _norm_text(s)
    return re.sub(r"[^a-z0-9 ]", "", s)


def _line_heading_like(line):
    """Section titles and doc titles look like this; body sentences don't.

    A 'Project X' mention found in a heading (e.g. '1. Project Overview') names
    a section, not the entity a claim is about, so it must not act as a strong
    subject that blocks cross-document pairing.
    """
    s = (line or "").strip()
    return bool(s) and len(s) < 80 and not s.endswith((".", "?", "!", ":", ";", ","))


def _match_strong(text, m):
    """A subject match is strong only when it sits in body text, not a heading."""
    ls = text.rfind("\n", 0, m.start()) + 1
    le = text.find("\n", m.end())
    return not _line_heading_like(text[ls:le if le >= 0 else len(text)])


def _scan_subject(text):
    """Find a 'Project X' mention. Returns (subject, strong); strong only when
    the mention sits in body text rather than a heading/title line."""
    text = text or ""
    first, strong_hit = None, None
    for m in _SUBJECT_RE.finditer(text):
        subj = "Project " + m.group(1)
        if first is None:
            first = subj
        if _match_strong(text, m):
            strong_hit = subj
            break
    if strong_hit is not None:
        return strong_hit, True
    if first is not None:
        return first, False
    return None, False


def _subject_of(chunk, doc_subject="", doc_strong=False):
    """Return (subject, strong). Headings and filename fallbacks are weak."""
    subj, strong = _scan_subject(chunk.get("text") or "")
    if subj is not None:
        return subj, strong
    if doc_subject:
        return doc_subject, doc_strong
    sec = (chunk.get("section") or "").strip()
    if sec and len(sec) < 80:
        return sec, False
    return (chunk.get("filename") or "").rsplit(".", 1)[0].replace("_", " "), False


# A date/amount keyword that introduces a DOCUMENT TITLE ("Production Release
# Memo", "Steering Committee Minutes") is a reference to another document, not
# an assertion about the value next to it. Such matches must not attribute.
_TITLE_WORDS = {"memo", "memorandum", "document", "report", "minutes"}


def _best_match(sentence, value_type):
    """Best (value, attribute-key) pairing in a sentence.

    Attribute = first keyword in list order (= specificity: "pilot" outranks
    the generic "go live"). Value = the normalizable match nearest an
    occurrence of that keyword, so a table cell can't steal the claim from a
    nearby assertion. Title references ("Production Release Memo") are never
    assertions. Returns ((raw, value_norm), canonical_attr) or (None, None).
    """
    low = sentence.lower()
    starts = [m.start() for m in re.finditer(r"[A-Za-z0-9]+", sentence)]

    def tok_idx(pos):
        n = 0
        for s in starts:
            if s < pos:
                n += 1
            else:
                break
        return n

    for key in TYPE_KEYS[value_type]:
        occs = []
        for km in re.finditer(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", low):
            nxt = re.match(r"\s*([a-z]+)", low[km.end():])
            if nxt and nxt.group(1) in _TITLE_WORDS:
                continue  # title reference, not an assertion
            occs.append(km.start())
        if not occs:
            continue
        best = None  # (distance, position, raw, norm)
        for pat in _PATTERNS[value_type]:
            for vm in pat.finditer(sentence):
                raw = vm.group(0).strip().rstrip(",;:")
                vnorm = _typed_value(raw, value_type)
                if vnorm is None:
                    continue  # unsafe to normalize: skip, do not guess
                vi = tok_idx(vm.start())
                d = min(abs(vi - tok_idx(o)) for o in occs)
                if best is None or (d, vm.start()) < (best[0], best[1]):
                    best = (d, vm.start(), raw, vnorm)
        if best:
            return (best[2], best[3]), _ATTR_ALIASES.get(key, key)
    return None, None


def _qualifiers_of(sentence):
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    return sorted(QUALIFIERS & words)


def _period_of(sentence):
    return sorted(set(_YEAR_RE.findall(sentence)))


# Revision-framed sentences discuss superseded values, not live assertions
# ("revised from X to Y", "updated ... original ..."). The "from"/past marker
# is required so plain current statements ("the revised budget is ...") stay live.
_REV_VERB = re.compile(r"\b(revis\w*|updat\w*|chang\w*|mov\w*|amend\w*|supersed\w*)\b", re.IGNORECASE)
_PAST_REF = re.compile(r"\bfrom\b|\boriginally\b|\bpreviously\b|\bformerly\b|\bprior\b|\bold\b|\bearlier\b|\boriginal\b",
                       re.IGNORECASE)


def _historical_context(sentence):
    """True when the sentence frames its value as superseded history."""
    return bool(_REV_VERB.search(sentence or "") and _PAST_REF.search(sentence or ""))


def _typed_value(raw, value_type):
    if value_type == "date":
        v = normalize_date(raw)
        return ("D:" + v) if v else None
    if value_type == "amount":
        v = normalize_amount(raw)
        return ("A:%s:%s" % (v[0], v[1])) if v else None
    if value_type == "percentage":
        v = normalize_percentage(raw)
        return ("P:%s" % v) if v is not None else None
    if value_type == "version":
        v = normalize_version(raw)
        return ("V:%s" % (".".join(map(str, v)),)) if v else None
    if value_type == "quantity":
        v = normalize_quantity(raw)
        return ("Q:%s:%s" % (v[0], v[1])) if v else None
    return None


def doc_subject_map(chunks):
    """Map doc_id -> (subject, strong), scanned across ALL chunks of that document.

    A passage about 'the delivery timeline' belongs to the project named
    anywhere in its own document, not just in the retrieved siblings.
    Strength is kept so heading-only mentions don't block cross-doc pairing.
    """
    out = {}
    for ch in chunks or []:
        subj, strong = _scan_subject(ch.get("text") or "")
        if subj is None:
            continue
        prev = out.get(ch.get("doc_id"))
        if prev is None or (strong and not prev[1]):
            out[ch["doc_id"]] = (subj, strong)
    return out


def doc_subjects_for(chunks):
    """Map doc_id -> 'Project X' scanned across ALL chunks of that document.

    A passage about 'the delivery timeline' belongs to the project named
    anywhere in its own document, not just in the retrieved siblings.
    """
    return {d: s for d, (s, _) in doc_subject_map(chunks).items()}


def extract_claims(evidence, doc_subjects=None):
    """Build claim dicts from retrieved evidence only. Never touches the corpus."""
    # Document-level subjects first: a passage about "the delivery timeline"
    # belongs to the project named elsewhere in the same document.
    # Values may be plain subjects (trusted, as before) or (subject, strong) pairs.
    doc_subjects = dict(doc_subjects or {})
    subs = {}
    for d, v in doc_subjects.items():
        subs[d] = (v[0], bool(v[1])) if isinstance(v, (tuple, list)) else (v, True)
    for ev in evidence:
        if ev.get("doc_id") in subs:
            continue
        subj, strong = _scan_subject(ev.get("text") or "")
        if subj is not None:
            subs[ev["doc_id"]] = (subj, strong)
    claims, n, seen_claims = [], 0, set()

    def emit(ev, seg, vtype):
        """Extract at most one claim of vtype from seg. False when Attributable
        nothing (or a duplicate); qualifiers/period/quote come from the segment."""
        nonlocal n
        found, attr = _best_match(seg, vtype)
        if not found or not attr:
            return False
        raw, vnorm = found
        periods = _period_of(seg) if vtype in ("amount", "quantity", "percentage", "text") else []
        quals = _qualifiers_of(seg)
        dup = (ev.get("doc_id"), _norm_key(attr), vnorm, tuple(periods), tuple(quals))
        if dup in seen_claims:
            return False  # same document already asserts this; keep the first
        seen_claims.add(dup)
        n += 1
        subj, strong = _subject_of(ev, *subs.get(ev.get("doc_id"), ("", False)))
        claims.append({
            "claim_id": "C%d" % n, "citation_id": ev.get("sid"),
            "doc_id": ev.get("doc_id"), "filename": ev.get("filename"),
            "page": ev.get("page"), "section": ev.get("section") or "",
            "subject": subj, "subject_strength": "strong" if strong else "weak",
            "attribute": attr,
            "value": raw, "value_norm": vnorm, "value_type": vtype,
            "qualifiers": quals, "historical": _historical_context(seg),
            "period": periods,
            "source_text": seg,
        })
        return True

    for ev in evidence:
        text = ev.get("text") or ""
        made = 0
        for sent in _SENT_SPLIT.split(text):
            sent = sent.strip()
            if len(sent) < 12 or made >= MAX_CLAIMS_PER_EVIDENCE:
                continue
            # Table-like sentences merge many values; extract per row so a table
            # cell can't steal the claim. The whole sentence runs only for types
            # no row could attribute (e.g. wrapped body text split mid-sentence).
            units = [sent]
            if "\n" in sent:
                rows = [r.strip() for r in sent.split("\n") if len(r.strip()) >= 12]
                if len(rows) > 1:
                    units = rows
            row_typed = set()
            if len(units) > 1:
                for seg in units:
                    if made >= MAX_CLAIMS_PER_EVIDENCE:
                        break
                    for vtype in _PATTERNS:
                        if emit(ev, seg, vtype):
                            row_typed.add(vtype)
                            made += 1
                for vtype in _PATTERNS:
                    if vtype not in row_typed and made < MAX_CLAIMS_PER_EVIDENCE:
                        if emit(ev, sent, vtype):
                            made += 1
            else:
                for vtype in _PATTERNS:
                    if emit(ev, sent, vtype):
                        made += 1
    return claims


# ---------------------------------------------------------------- candidate pairing

def _same_subject(a, b):
    sa, sb = _norm_key(a.get("subject")), _norm_key(b.get("subject"))
    return bool(sa and sb) and sa == sb


def _subjects_compatible(a, b):
    """Same subject required — with one narrow exception: when BOTH subjects
    are weak (section headings, doc titles, filename fallbacks), neither is a
    positive identification, so neither may veto cross-document pairing.
    A positively identified (strong, explicit body mention) subject still
    vetoes anything different — including unknown ones.
    """
    sa, sb = _norm_key(a.get("subject")), _norm_key(b.get("subject"))
    if sa == sb:
        return bool(sa)
    if not sa or not sb:
        return True  # unknown subject can't veto
    wa = a.get("subject_strength", "strong") != "strong"
    wb = b.get("subject_strength", "strong") != "strong"
    return wa and wb


def _same_attribute(a, b):
    return _norm_key(a.get("attribute")) == _norm_key(b.get("attribute"))


def pair_candidates(claims):
    """Claims -> candidate pairs: same subject+attribute+type, differing values.

    Qualifiers NEVER auto-confirm or auto-reject; differing values always go
    to the verifier. Identical normalized values are compatible (no pair).
    """
    cands = []
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            a, b = claims[i], claims[j]
            if a["citation_id"] == b["citation_id"]:
                continue  # same passage cannot conflict with itself
            if a["value_type"] != b["value_type"]:
                continue
            if a["value_type"] == "amount":
                # No FX conversion: different currencies are never contradictions.
                ca = a["value_norm"].split(":")[1]
                cb = b["value_norm"].split(":")[1]
                if ca != cb:
                    continue
            if not _subjects_compatible(a, b) or not _same_attribute(a, b):
                continue
            if a.get("doc_id") == b.get("doc_id") and (a.get("historical") or b.get("historical")):
                continue  # same-document revision narrative, not a live contradiction
            if a.get("period") != b.get("period"):
                continue  # different time periods (e.g. 2026 vs 2027 revenue)
            if a["value_norm"] == b["value_norm"]:
                continue  # equivalent forms: compatible
            cands.append({"claim_a": a, "claim_b": b})
    return cands


def filter_question_relevant(candidates, question):
    """Keep only pairs about what the question asks. A shown conflict must
    concern the question's subject matter, not an unrelated field that merely
    co-occurs in the retrieved passages."""
    qtokens = set(re.findall(r"[a-z0-9]+", (question or "").lower())) - _STOPWORDS
    if not qtokens:
        return []
    kept = []
    for pair in candidates:
        words = set()
        for c in (pair["claim_a"], pair["claim_b"]):
            words |= set(re.findall(r"[a-z0-9]+", _norm_key(c.get("attribute"))))
            words |= set(re.findall(r"[a-z0-9]+", _norm_key(c.get("subject"))))
        words -= _GENERIC_TOKENS
        if words & qtokens:
            kept.append(pair)
    return kept


# Qualifier sets that suggest DIFFERENT contexts (forecast vs approved) rather
# than a disagreement about the same current fact. Used for deterministic
# verdicts where no LLM is available (e.g. document comparison).
_CONTRAST_EARLY = {"forecast", "draft", "proposed", "previous"}
_CONTRAST_LATE = {"approved", "actual", "final", "current"}


def qualifiers_contrast(qa, qb):
    """True when the two qualifier sets look like different contexts
    (one early-stage, one settled) rather than rival claims."""
    sa, sb = set(qa or []), set(qb or [])
    return bool((sa & _CONTRAST_EARLY and sb & _CONTRAST_LATE) or
                (sb & _CONTRAST_EARLY and sa & _CONTRAST_LATE))


def compare_claims(claims, by_cid, doc_texts=None):
    """Deterministic document comparison over extracted claims.

    Reuses extraction/normalization/pairing rules (no second engine, no LLM).
    Returns rows: compatible (same normalized value), conflict (incompatible
    values, same subject/attribute/period/type), needs_review (qualifier-
    contrasted), info (differing values that pairing rules refuse to join,
    e.g. different periods or currencies). Never manufactures conflicts.
    """
    doc_texts = doc_texts or {}
    groups = {}
    for cl in claims:
        key = (_norm_key(cl.get("attribute")), cl.get("value_type"))
        groups.setdefault(key, []).append(cl)
    rows = []
    for (attr, vtype), g in groups.items():
        # Partition into subject-compatible blocks (deterministic order), so
        # claims about the same matter compare even when per-doc fallback
        # subjects differ ("Overview" vs "Update"), while two differing STRONG
        # subjects still never merge.
        ordered = sorted(g, key=lambda x: (x["doc_id"], x["citation_id"]))
        blocks = []
        for c in ordered:
            placed = False
            for bl in blocks:
                if all(_subjects_compatible(c, m) for m in bl):
                    bl.append(c)
                    placed = True
                    break
            if not placed:
                blocks.append([c])
        for b in blocks:
            _append_compare_row(rows, b, vtype, by_cid, doc_texts)
    order = {"conflict": 0, "needs_review": 1, "info": 2, "compatible": 3}
    rows.sort(key=lambda r: (order[r["verdict"]], r["subject"], r["attribute"]))
    return rows


def _append_compare_row(rows, b, vtype, by_cid, doc_texts):
    docs = sorted({c["doc_id"] for c in b})
    if len(docs) < 2:
        return  # single-document rows are not comparisons
    norms = {c["value_norm"] for c in b}
    base = {"subject": b[0]["subject"], "attribute": b[0]["attribute"], "value_type": vtype,
            "entries": [{"doc_id": c["doc_id"], "filename": c["filename"], "page": c["page"],
                         "chunk_id": c["citation_id"], "method": by_cid.get(c["citation_id"], {}).get("method", "text"),
                         "value": c["value"], "value_norm": c["value_norm"],
                         "qualifiers": c["qualifiers"]} for c in b],
            "hint": None}
    if len(norms) == 1:
        rows.append(dict(base, verdict="compatible",
                         reason="All documents state the same normalized value."))
        return
    pairs = pair_candidates(b)
    if not pairs:
        rows.append(dict(base, verdict="info",
                         reason="Values differ in form but pairing rules treat them as different "
                                "contexts (e.g. different periods or currencies), not a conflict."))
        return
    if any(qualifiers_contrast(p["claim_a"]["qualifiers"], p["claim_b"]["qualifiers"]) for p in pairs):
        rows.append(dict(base, verdict="needs_review",
                         reason="Differing values with contrasting qualifiers (e.g. forecast vs approved); "
                                "human review needed to decide compatibility."))
        return
    row = dict(base, verdict="conflict",
               reason="Incompatible normalized values for the same subject and attribute.")
    # Reuse the exact hint behavior: build a minimal conflict for the first pair.
    p = pairs[0]
    pseudo = {"citation_a": p["claim_a"]["citation_id"], "citation_b": p["claim_b"]["citation_id"],
              "likely_newer": None, "newer_reason": None, "_a": p["claim_a"], "_b": p["claim_b"]}
    attach_hints([pseudo], by_cid, doc_texts)
    if pseudo["likely_newer"]:
        winner = next(e for e in row["entries"] if e["chunk_id"] == pseudo["likely_newer"])
        row["hint"] = {"doc_id": winner["doc_id"], "filename": winner["filename"],
                       "reason": pseudo["newer_reason"]}
    rows.append(row)


# ---------------------------------------------------------------- LLM verification

VERIFY_SYSTEM = """You classify whether two extracted claims are actually incompatible.
Return JSON only: {"classification": "conflict" | "compatible" | "insufficient_context", "reason": "...", "claim_a": "...", "claim_b": "..."}.
Rules:
- Inspect ONLY the two supplied claims and their source context. Do not use outside knowledge.
- Qualifiers matter: forecast vs actual, draft vs approved, historical vs current, or different time periods may be COMPATIBLE.
- Different subjects are never conflicts. Additional detail that does not contradict is not a conflict.
- Only incompatible claims about the SAME subject and attribute are "conflict".
- When unsure, use "insufficient_context", never guess.
- Do NOT invent a resolution or pick a winner."""


def _verify_prompt(pair):
    def fmt(c):
        return ("claim %s [%s] subject=%r attribute=%r value=%r (normalized %r, type %s) "
                "qualifiers=%s context=%r" % (
                    c["claim_id"], c["citation_id"], c["subject"], c["attribute"],
                    c["value"], c["value_norm"], c["value_type"],
                    ",".join(c["qualifiers"]) or "none", c["source_text"]))
    return "Claim 1: %s\nClaim 2: %s" % (fmt(pair["claim_a"]), fmt(pair["claim_b"]))


def verify_candidates(candidates, llm_fn):
    """Classify each candidate via the LLM. Transport/parse failures of the
    verifier fail OPEN to 'unresolved' (show both claims side-by-side) rather
    than silently dropping a possible conflict. Returns list of conflict dicts."""
    out = []
    for pair in candidates:
        a, b = pair["claim_a"], pair["claim_b"]
        classification, reason = "unresolved", "Verifier inconclusive; shown side-by-side."
        try:
            raw = llm_fn(VERIFY_SYSTEM, _verify_prompt(pair)) or {}
            if raw.get("classification") in ("conflict", "compatible", "insufficient_context"):
                classification = raw["classification"]
                reason = raw.get("reason") or reason
            else:
                log.warning("Verifier returned unexpected shape; keeping candidate unresolved.")
        except Exception as e:
            log.warning("Verifier call failed (%s); keeping candidate unresolved.", e)
        if classification == "compatible":
            continue
        field = "%s: %s" % (a["subject"], a["attribute"]) if a["subject"] else a["attribute"]
        out.append({
            "field": field, "claim_a": a["value"], "citation_a": a["citation_id"],
            "claim_b": b["value"], "citation_b": b["citation_id"],
            "conflict_type": a["value_type"], "resolution": "unresolved",
            "reason": reason, "likely_newer": None, "newer_reason": None,
            "_a": a, "_b": b,
        })
    return out


# ---------------------------------------------------------------- resolution hints (hints only, never verdicts)

_HINT_MARKERS = [(re.compile(r"supersed\w*", re.IGNORECASE), 3, "explicitly states that it supersedes the earlier document"),
                 (re.compile(r"\brevis\w*", re.IGNORECASE), 2, "describes itself as revised"),
                 (re.compile(r"\bupdat\w*", re.IGNORECASE), 1, "describes itself as updated")]


def attach_hints(conflicts, by_sid, doc_texts=None):
    """Set likely_newer from explicit textual signals only. Signals are read
    from the full source DOCUMENT (the marker often lives in a sibling chunk,
    e.g. a 'Status: this document supersedes...' paragraph). Mutates in place."""
    doc_texts = doc_texts or {}
    for c in conflicts:
        scores = {}
        for side in ("a", "b"):
            claim = c["_" + side]
            cite = claim["citation_id"]
            chunk_text = (by_sid.get(cite, {}).get("text") or "")
            doc_text = doc_texts.get(claim.get("doc_id"), "")
            text = chunk_text + "\n" + doc_text
            score, why, sent = 0, None, None
            for pat, pts, label in _HINT_MARKERS:
                m = pat.search(text)
                if m:
                    score += pts
                    if why is None:
                        why, sent = label, _marker_sentence(text, m)
            scores[side] = (score, why, sent)
        (sa, wa, snt_a), (sb, wb, snt_b) = scores["a"], scores["b"]
        if sa != sb:
            winner = "a" if sa > sb else "b"
            why, sent = (wa, snt_a) if winner == "a" else (wb, snt_b)
            basis = (" explicitly states %r" % sent) if sent else (" %s" % why)
            c["likely_newer"] = c["citation_" + winner]
            c["newer_reason"] = ("Likely more current because the %s source%s "
                                 "(hint only, verify with the owner)." % (c["citation_" + winner], basis))
    return conflicts


def _marker_sentence(text, match):
    start = text.rfind(".", 0, match.start()) + 1
    end = text.find(".", match.end())
    sent = text[start:end if end >= 0 else match.end() + 120].strip()
    return (sent[:200] + "...") if len(sent) > 200 else sent
