"""Cross-document conflict regression: revision-narrative pattern.

Generic fixtures (HarborLink) reproduce the reported failure mode without
hardcoding any demo document, date, or question:
- doc A asserts a current value;
- doc B revises it, quoting the old value inside a revision narrative
  ("revised from X to Y") and asserting the new value elsewhere.

The engine must report doc A -> old vs doc B -> new (CONFLICTING), never
present doc B's historical reference as a live claim, keep the supersedes
hint, and stay quiet on unrelated questions.
"""
import os
import re

import main
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = TestClient(main.app)
IDS = {}
_ORIG = main.llm_with_system


def _upload():
    if IDS:
        return IDS
    files = [("files", (n.split("/")[-1], open(os.path.join(ROOT, n), "rb")))
             for n in ("tests/fixtures/harbor_memo.txt", "tests/fixtures/harbor_update.txt")]
    r = client.post("/documents", files=files)
    assert r.status_code == 200, r.text
    IDS.update({d["filename"]: d["id"] for d in r.json()})
    return IDS


def _stub(system, user):
    if "classify whether two extracted claims" in system:
        return {"classification": "conflict", "reason": "stub: values differ."}
    mq = re.search(r"^Question: (.*)$", user, re.M)
    question = mq.group(1) if mq else ""
    sids = re.findall(r"\[(S\d+)\]", user)
    if "VERIFIED CONFLICTS" in user:
        return {"answer_status": "conflicting", "answer": "stub conflict.", "confidence_label": "high",
                "confidence_reason": "stub.", "citations": sids[:2], "conflicts": [], "missing_information": []}
    if question == "When did the pilot launch?":
        return {"answer_status": "supported", "answer": "stub: 20 April.", "confidence_label": "high",
                "confidence_reason": "stub.", "citations": sids[:1], "conflicts": [], "missing_information": []}
    return {"answer_status": "insufficient_evidence", "answer": "Not found in the uploaded documents.",
            "confidence_label": "high", "confidence_reason": "stub.", "citations": [],
            "conflicts": [], "missing_information": ["requested info"]}


def _ask(question):
    main.llm_with_system = _stub
    try:
        r = client.post("/investigate", json={"question": question})
    finally:
        main.llm_with_system = _ORIG
    assert r.status_code == 200, r.text
    return r.json()


def _by_filename(d, filename):
    for c in d["citations"]:
        if c["filename"] == filename:
            return c["doc_id"]
    raise AssertionError("no citation from %s" % filename)


def test_01_release_conflicting_cross_document():
    _upload()
    d = _ask("What is the production release date?")
    assert d["status"] == "conflicting"
    assert d["conflicts"], "expected at least one conflict pair"
    pair = None
    for c in d["conflicts"]:
        da = next(x for x in d["citations"] if x["id"] == c["a"]["cite"])["filename"]
        db = next(x for x in d["citations"] if x["id"] == c["b"]["cite"])["filename"]
        if {da, db} == {"harbor_memo.txt", "harbor_update.txt"}:
            pair = (c, da, db)
            break
    assert pair, "need a cross-document conflict pair"
    c, da, db = pair
    old = c["a"] if da == "harbor_memo.txt" else c["b"]
    new = c["b"] if da == "harbor_memo.txt" else c["a"]
    assert "12 May 2026" in old["claim"] and "28 May 2026" in new["claim"]
    assert c.get("likely_newer") == new["cite"], "revised source must be hinted"
    assert "supersede" in (c.get("newer_reason") or "") or "revis" in (c.get("newer_reason") or "")


def test_02_budget_conflicting():
    _upload()
    d = _ask("What is the project budget?")
    assert d["status"] == "conflicting"
    blob = " ".join([c["a"]["claim"] + c["b"]["claim"] for c in d["conflicts"]])
    assert "40,000" in blob and "47,500" in blob


def test_03_pilot_supported_compatible():
    _upload()
    d = _ask("When did the pilot launch?")
    assert d["status"] == "supported"
    assert d["conflicts"] == []


def test_04_unknown_insufficient():
    _upload()
    d = _ask("What security certification does HarborLink have?")
    assert d["status"] == "insufficient_evidence"
    assert d["conflicts"] == []


def test_05_retrieval_represents_both_docs():
    _upload()
    d = _ask("What is the production release date?")
    docs = {x["doc_id"] for x in d["investigation"]["documents_considered"]}
    assert docs >= {IDS["harbor_memo.txt"], IDS["harbor_update.txt"]}


def test_06_compare_fixtures():
    _upload()
    r = client.post("/compare", json={"doc_ids": [IDS["harbor_memo.txt"], IDS["harbor_update.txt"]]})
    assert r.status_code == 200, r.text
    d = r.json()
    rel = [x for x in d["rows"] if x["attribute"] == "release date"]
    assert len(rel) == 1 and rel[0]["verdict"] == "conflict"
    got = {e["value"] for e in rel[0]["entries"]}
    assert {"12 May 2026", "28 May 2026"} <= got
    assert {e["doc_id"] for e in rel[0]["entries"]} >= {IDS["harbor_memo.txt"], IDS["harbor_update.txt"]}
    assert rel[0]["hint"] and rel[0]["hint"]["filename"] == "harbor_update.txt"
    bud = [x for x in d["rows"] if x["attribute"] == "budget"]
    assert bud and all(x["verdict"] == "conflict" for x in bud)
    pilot = [x for x in d["rows"] if x["attribute"] == "pilot date"]
    assert pilot and all(x["verdict"] == "compatible" for x in pilot)


def test_07_heading_subjects_are_weak():
    import conflicts as C
    assert C._subject_of({"text": "1. Project Overview\nNexoraPay is great.", "section": "",
                          "filename": "x.txt"}) == ("Project Overview", False)
    assert C._subject_of({"text": "Project Atlas release date is 30 June 2027.", "section": "",
                          "filename": "x.txt"}) == ("Project Atlas", True)
    assert C.doc_subject_map([{"doc_id": "d", "text": "1. Project Overview\nBody here."}]) == \
        {"d": ("Project Overview", False)}


def test_08_explicit_subjects_still_block():
    import conflicts as C

    def ev(sid, text, doc):
        return {"sid": sid, "doc_id": doc, "filename": doc + ".txt", "page": 1,
                "section": "", "method": "text", "text": text}

    cs = C.extract_claims([ev("S1", "Project Alpha release date is 30 June 2027.", "a"),
                           ev("S2", "Project Beta release date is 15 July 2027.", "b")])
    assert all(c["subject_strength"] == "strong" for c in cs)
    assert C.pair_candidates(cs) == []


def test_09_historical_same_doc_skipped():
    import conflicts as C

    def ev(sid, text, doc):
        return {"sid": sid, "doc_id": doc, "filename": doc + ".txt", "page": 1,
                "section": "", "method": "text", "text": text}

    rev = "The production release date was revised from 12 May 2026 to 28 May 2026."
    cur = "The production release date is now 28 May 2026."
    same = C.extract_claims([ev("S1", rev, "a"), ev("S2", cur, "a")])
    assert [c["value"] for c in same] == ["12 May 2026", "28 May 2026"]
    assert same[0]["historical"] and not same[1]["historical"]
    # same-document revision narrative: no candidate pair
    assert C.pair_candidates(same) == []
    # ...but across documents the same values still pair for the verifier
    cross = C.extract_claims([ev("S1", rev, "a"), ev("S2", cur, "b")])
    assert len(C.pair_candidates(cross)) == 1
