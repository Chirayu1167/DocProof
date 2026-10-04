"""Final-phase tests: investigation_summary, decision_reason, claim/citation
provenance, hints, compare regression, trail/strength regression, safety."""
import json
import os
import re

import pytest

import main
from fastapi.testclient import TestClient
from fastapi import HTTPException

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
client = TestClient(main.app)
IDS = {}
_ORIG_LLM = main.llm_with_system


@pytest.fixture(autouse=True)
def _restore_llm():
    yield
    main.llm_with_system = _ORIG_LLM


def _stub(system, user):
    if "classify whether two extracted claims" in system:
        return {"classification": "conflict", "reason": "stub: values differ."}
    mq = re.search(r"^Question: (.*)$", user, re.M)
    question = mq.group(1) if mq else ""
    sids = re.findall(r"\[(S\d+)\]", user)
    if "VERIFIED CONFLICTS" in user:
        return {"answer_status": "conflicting", "answer": "stub conflict.", "confidence_label": "high",
                "confidence_reason": "stub.", "citations": sids[:2], "conflicts": [], "missing_information": []}
    if question == "Where will the pilot go live?":
        return {"answer_status": "supported", "answer": "stub: Pune campus.", "confidence_label": "high",
                "confidence_reason": "stub.", "citations": sids[:1], "conflicts": [], "missing_information": []}
    return {"answer_status": "insufficient_evidence", "answer": "Not found in the uploaded documents.",
            "confidence_label": "high", "confidence_reason": "stub.", "citations": [],
            "conflicts": [], "missing_information": ["requested info"]}


def _upload_demos():
    if IDS:
        return IDS
    files = [("files", (n, open(os.path.join(ROOT, n), "rb")))
             for n in ("1_vendor_proposal.txt", "2_steering_minutes.txt", "3_revised_schedule.txt")]
    r = client.post("/documents", files=files)
    assert r.status_code == 200, r.text
    IDS.update({d["filename"]: d["id"] for d in r.json()})
    return IDS


def _ask(question):
    main.llm_with_system = _stub
    r = client.post("/investigate", json={"question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_01_summary_conflicting():
    _upload_demos()
    d = _ask("What is the production release date?")
    assert d["status"] == "conflicting"
    s = d["investigation_summary"]
    assert set(s) == {"documents_examined", "passages_retrieved", "relevant_passages",
                      "claims_extracted", "claims_compared", "conflicts_found",
                      "supporting_sources", "retrieval", "status"}
    assert s["documents_examined"] >= 2
    assert s["passages_retrieved"] >= 2
    assert s["relevant_passages"] >= 1
    assert s["claims_extracted"] >= 2
    assert s["claims_compared"] >= 2
    assert s["conflicts_found"] >= 1
    assert s["supporting_sources"] >= 2
    assert s["retrieval"]["bm25"] is True and s["retrieval"]["fusion"] == "RRF"
    assert s["retrieval"]["semantic"] in ("local", "openai", "off")
    assert s["status"] == "conflicting"


def test_02_decision_reason_conflicting():
    _upload_demos()
    d = _ask("What is the production release date?")
    dr = d["decision_reason"]
    assert dr["type"] == "conflict"
    assert "different values" in dr["summary"]
    blob = " ".join(dr["details"])
    assert "vendor proposal" in blob and "revised schedule" in blob
    assert "30 June 2027" in blob and "15 July 2027" in blob
    assert any("hint only" in x for x in dr["details"])
    assert dr["details"][-1] == "The system does not automatically select one value."


def test_03_decision_reason_supported():
    _upload_demos()
    d = _ask("Where will the pilot go live?")
    assert d["status"] == "supported"
    dr = d["decision_reason"]
    assert dr["type"] == "support"
    assert "validated citation" in dr["summary"]
    assert dr["details"] and all("(p. " in x for x in dr["details"] if not x.startswith("..."))
    assert d["investigation_summary"]["status"] == "supported"


def test_04_decision_reason_insufficient():
    _upload_demos()
    d = _ask("Who owns the trademark?")
    assert d["status"] == "insufficient_evidence"
    dr = d["decision_reason"]
    assert dr["type"] == "insufficient"
    assert dr["summary"] == "The uploaded documents do not establish the answer."
    assert dr["details"]
    s = d["investigation_summary"]
    assert s["conflicts_found"] == 0 and s["status"] == "insufficient_evidence"


def test_05_claims_citations_provenance():
    _upload_demos()
    d = _ask("What is the production release date?")
    ev_ids = {c["citation_id"] for c in d["investigation"]["claims"]}
    assert ev_ids, "trail must expose claims"
    for c in d["investigation"]["claims"]:
        assert c["citation_id"].startswith("S"), "trail citation ids are backend S-ids"
        assert set(c) >= {"claim_id", "citation_id", "filename", "page", "subject",
                          "attribute", "value", "value_norm", "value_type", "qualifiers"}
    for c in d["citations"]:
        assert c["id"].startswith("S")
        assert c["doc_id"] in main.DOCS, "citation doc must exist"
    chunks = [c["text"] for c in main.CHUNKS]
    for cf in d["conflicts"]:
        for side in ("a", "b"):
            q = main.norm(cf[side]["quote"])
            assert any(q in main.norm(t) for t in chunks), \
                "conflict quote must occur verbatim in a stored chunk"


def test_06_hint_preserved():
    _upload_demos()
    d = _ask("What is the production release date?")
    hints = [c for c in d["conflicts"] if c.get("likely_newer")]
    assert hints, "supersedes hint must survive"
    assert any("supersede" in (c.get("newer_reason") or "") or "revis" in (c.get("newer_reason") or "")
               for c in hints)


def test_07_compare_regression():
    _upload_demos()
    r = client.post("/compare", json={"doc_ids": [IDS["1_vendor_proposal.txt"],
                                                   IDS["3_revised_schedule.txt"]]})
    assert r.status_code == 200, r.text
    d = r.json()
    rows = [x for x in d["rows"] if x["attribute"] == "release date"]
    assert len(rows) == 1 and rows[0]["verdict"] == "conflict"
    assert rows[0]["hint"] and rows[0]["hint"]["filename"] == "3_revised_schedule.txt"
    for e in rows[0]["entries"]:
        assert {"doc_id", "filename", "page", "chunk_id", "value", "value_norm"} <= set(e)
        assert e["doc_id"] in main.DOCS


def test_08_trail_strength_regression():
    _upload_demos()
    for q, st in (("What is the production release date?", "conflicting"),
                  ("Where will the pilot go live?", "supported"),
                  ("Who owns the trademark?", "insufficient_evidence")):
        d = _ask(q)
        assert d["status"] == st
        assert isinstance(d["investigation"]["checks"], list) and d["investigation"]["checks"]
        assert d["evidence_strength"]["level"] in ("STRONG", "MODERATE", "INSUFFICIENT", "CONFLICTING")
        assert d["evidence_strength"]["reasons"]


def test_09_safety():
    _upload_demos()
    assert client.post("/investigate", json={"question": "   "}).status_code == 400
    r = client.post("/documents", files=[("files", ("evil.exe", b"MZ fake"))])
    assert r.status_code == 400
    assert client.get("/documents/nope00/pages/1").status_code == 404
    with pytest.raises(HTTPException) as ei:
        main.parse_llm_json("not json at all {{{")
    assert ei.value.status_code == 502
    # no filesystem paths leak through investigation or compare payloads
    d = _ask("What is the production release date?")
    blob = json.dumps({"c": d["citations"], "i": d["investigation"], "s": d["investigation_summary"]})
    assert "uploads" not in blob and "D:\\" not in blob and ".env" not in blob.lower()
    r = client.post("/compare", json={"doc_ids": [IDS["1_vendor_proposal.txt"],
                                                   IDS["3_revised_schedule.txt"]]})
    blob2 = r.text
    assert "uploads" not in blob2 and "D:\\" not in blob2


def test_10_ocr_actionable_error():
    fx = os.path.join(ROOT, "tests", "fixtures", "scan_memo.png")
    assert os.path.exists(fx), "OCR fixture must exist"
    r = client.post("/documents", files=[("files", ("scan_memo.png", open(fx, "rb")))])
    # No Tesseract binary in this environment -> must fail with an actionable message, never a crash
    assert r.status_code == 400
    assert "OCR" in r.json()["detail"]
