"""Evaluation runner: `python -m tests.run_eval`.

Runs tests/tests.json through the REAL backend pipeline (upload + retrieve +
deterministic conflicts + answer assembly) and writes tests/report.json.

Modes (EVAL_MODE env or --mode flag):
  stubbed (default) - LLM calls are replaced by deterministic test doubles
      with documented semantics (see STUB_NOTES). No network, no cost.
  live - real LLM via main.llm_with_system. Requires LLM_API_KEY.

The report ALWAYS labels which mode produced it. Stubbed results are never
presented as model results.
"""
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STUB_NOTES = (
    "STUBBED evaluation: the answer-generator returns canned responses derived "
    "from each test's expectations using only citation IDs present in the prompt; "
    "the conflict verifier returns 'compatible' only for qualifier-contrasted pairs "
    "(forecast/draft/proposed vs approved/actual/final/current), else 'conflict'. "
    "Retrieval, claim extraction, normalization, pairing, relevance filtering, "
    "hint attachment, citation validation and status enforcement are all REAL."
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import main
from fastapi.testclient import TestClient

try:
    import conflicts as CM
except Exception:
    CM = None

MANIFEST = ROOT / "tests" / "tests.json"
REPORT = ROOT / "tests" / "report.json"

CONTRAST_EARLY = {"forecast", "draft", "proposed", "previous"}
CONTRAST_LATE = {"approved", "actual", "final", "current"}


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ---------------------------------------------------------------- stubs
def stub_verifier(prompt):
    quals = re.findall(r"qualifiers=([^ )\]]+)", prompt)
    qa = set(quals[0].split(",")) if len(quals) > 0 and quals[0] != "none" else set()
    qb = set(quals[1].split(",")) if len(quals) > 1 and quals[1] != "none" else set()
    if (qa & CONTRAST_EARLY and qb & CONTRAST_LATE) or (qb & CONTRAST_EARLY and qa & CONTRAST_LATE):
        return {"classification": "compatible",
                "reason": "stub: qualifier-contrasted contexts (e.g. forecast vs approved).",
                "claim_a": "a", "claim_b": "b"}
    return {"classification": "conflict", "reason": "stub: same subject/attribute, different values.",
            "claim_a": "a", "claim_b": "b"}


def sid_blocks(prompt):
    """Parse the [Sn] evidence blocks of an answer prompt into {sid: text}."""
    parts = re.split(r"\[S(\d+)\]", prompt)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        out["S" + parts[i]] = parts[i + 1]
    return out


def make_answer_stub(q2test):
    def stub(system, user):
        if "classify whether two extracted claims" in system:
            return stub_verifier(user)
        test = None
        for q, t in q2test.items():
            if q in user:
                test, blocks = t, sid_blocks(user)
                break
        if test is None:
            return {"answer_status": "insufficient_evidence", "answer": "Not found in the uploaded documents.",
                    "confidence_label": "low", "confidence_reason": "stub: unknown question.",
                    "citations": [], "conflicts": [], "missing_information": ["context"]}
        sids = list(blocks)
        if (test.get("expected_status") == "conflicting" or
                "conflicting" in test.get("allowed_statuses", []) and test.get("expected_claims")) \
                or test.get("expected_claims") and test.get("expected_status") == "conflicting":
            claims = test.get("expected_claims", [])
            picked = []
            for cl in claims:
                hit = next((s for s in sids if norm(cl) in norm(blocks[s])), None)
                if hit and hit not in picked:
                    picked.append(hit)
            picked += [s for s in sids if s not in picked]
            n = max(test.get("min_citations", 2), 2)
            return {"answer_status": "conflicting",
                    "answer": "stub: conflicting claims: %s." % " vs ".join(claims),
                    "confidence_label": "high", "confidence_reason": "stub.",
                    "citations": picked[:n], "conflicts": [], "missing_information": []}
        if test.get("expected_status") == "supported":
            snips = [rc["snippet"] for rc in test.get("required_citations", [])]
            must = []
            for sn in snips:  # cover every required snippet, not just the first
                hit = next((s for s in sids if norm(sn) in norm(blocks[s])), None)
                if hit and hit not in must:
                    must.append(hit)
            picked = must + [s for s in sids if s not in must]
            n = max(test.get("min_citations", 1), len(must), 1)
            return {"answer_status": "supported",
                    "answer": "stub: supported by evidence: %s." % "; ".join(snips or ["see citations"]),
                    "confidence_label": "high", "confidence_reason": "stub.",
                    "citations": picked[:n], "conflicts": [], "missing_information": []}
        return {"answer_status": "insufficient_evidence",
                "answer": "Not found in the uploaded documents.",
                "confidence_label": "high", "confidence_reason": "stub: no supporting evidence.",
                "citations": [], "conflicts": [], "missing_information": ["requested information"]}
    return stub


# ---------------------------------------------------------------- checkers
def find_chunk(doc_id, resp_text):
    for ch in main.CHUNKS:
        if ch.get("doc_id") == doc_id and resp_text and resp_text in ch.get("text", ""):
            return ch
    return None


def check_citations(resp, sid_chunk):
    """Every returned citation must resolve to real backend evidence with a genuine quote."""
    problems, valid = [], 0
    for cit in resp.get("citations", []):
        sid = cit.get("id")
        chunk = sid_chunk.get(sid)
        if chunk is None:
            problems.append("%s: unknown citation id" % sid)
            continue
        doc = main.DOCS.get(cit.get("doc_id"))
        if doc is None or doc.get("filename") != cit.get("filename"):
            problems.append("%s: filename/doc mismatch" % sid)
            continue
        if not (1 <= cit.get("page", 0) <= doc.get("pages", 0)):
            problems.append("%s: page out of range" % sid)
            continue
        if not find_chunk(chunk["doc_id"], cit.get("text")):
            problems.append("%s: citation text not found in stored chunk" % sid)
            continue
        valid += 1
    for cf in resp.get("conflicts", []):
        for side in ("a", "b"):
            s = cf.get(side, {})
            chunk = sid_chunk.get(s.get("cite"))
            if chunk is None:
                problems.append("conflict %s: unknown citation id" % s.get("cite"))
            elif not find_chunk(chunk["doc_id"], s.get("quote")):
                problems.append("conflict %s: quote not found in stored chunk" % s.get("cite"))
            else:
                valid += 1
    total = len(resp.get("citations", [])) + 2 * len(resp.get("conflicts", []))
    return valid, total, problems


def claim_norms(text):
    """Extract normalized typed values mentioned in a claim string."""
    out = []
    if CM is None:
        return out
    for vtype, patterns in CM._PATTERNS.items():
        for pat in patterns:
            mm = pat.search(text or "")
            if mm:
                v = CM._typed_value(mm.group(0).strip().rstrip(",;:"), vtype)
                if v:
                    out.append(v)
                break
    return out


def run_case(client, test, sid_chunk_fn):
    t0 = time.perf_counter()
    try:
        r = client.post("/investigate", json={"question": test["question"]})
        resp = r.json() if r.status_code == 200 else {"_http": r.status_code, "_detail": r.text[:200]}
    except Exception as e:
        resp = {"_http": 0, "_detail": str(e)[:200]}
    dt_ms = (time.perf_counter() - t0) * 1000
    failures = []
    if "_http" in resp:
        return {"id": test["id"], "category": test["category"], "passed": False,
                "failures": ["HTTP %s: %s" % (resp["_http"], resp["_detail"])],
                "latency_ms": round(dt_ms, 1)}
    status = resp.get("status")
    exp = test.get("expected_status")
    if exp and status != exp:
        failures.append("status: expected %s, got %s" % (exp, status))
    if test.get("allowed_statuses") and status not in test["allowed_statuses"]:
        failures.append("status: expected one of %s, got %s" % (test["allowed_statuses"], status))
    if test.get("forbid_conflicts") and resp.get("conflicts"):
        failures.append("unexpected conflicts reported (%d)" % len(resp["conflicts"]))
    hay = json.dumps(resp.get("conflicts", [])) + resp.get("answer", "") + \
        "".join(c.get("text", "") for c in resp.get("citations", []))
    for cl in test.get("expected_claims", []):
        if norm(cl) not in norm(hay):
            failures.append("expected claim missing from response: %r" % cl)
    if test.get("expected_conflict_type"):
        if not any(c.get("conflict_type") == test["expected_conflict_type"] for c in resp.get("conflicts", [])):
            failures.append("no conflict of type %s" % test["expected_conflict_type"])
    if len(resp.get("citations", [])) < test.get("min_citations", 0):
        failures.append("citations: expected >= %d, got %d" % (test["min_citations"], len(resp["citations"])))
    for rc in test.get("required_citations", []):
        hit = any(c.get("filename") == rc["filename"] and c.get("page") == rc["page"]
                  and norm(rc["snippet"]) in norm(c.get("text", "")) for c in resp.get("citations", []))
        if not hit:
            failures.append("required citation missing: %s p.%s %r" % (rc["filename"], rc["page"], rc["snippet"]))
    for pair in test.get("forbidden_claim_values", []):
        want = set(pair)
        for cf in resp.get("conflicts", []):
            got = set(claim_norms(cf.get("a", {}).get("claim", "")) + claim_norms(cf.get("b", {}).get("claim", "")))
            if want <= got:
                failures.append("forbidden qualifier pair reported as conflict: %s" % (pair,))
    if test.get("require_hint"):
        if not any(c.get("likely_newer") and "supersede" in norm(c.get("newer_reason", ""))
                   for c in resp.get("conflicts", [])):
            failures.append("expected supersedes hint (likely_newer) missing")
    # state invariants
    if status == "conflicting" and not resp.get("conflicts"):
        failures.append("invariant: conflicting status exposes no conflicting evidence")
    if status == "supported" and not resp.get("citations"):
        failures.append("invariant: supported status has no supporting citation")
    if status == "insufficient_evidence" and resp.get("conflicts"):
        failures.append("invariant: insufficient_evidence must not manufacture conflicts")
    # citation validity (deterministic)
    sid_chunk = sid_chunk_fn(test["question"])
    valid, total, problems = check_citations(resp, sid_chunk)
    failures.extend("citation: " + p for p in problems)
    # retrieval recall ground truth
    r0 = time.perf_counter()
    ev, _, _ = main.retrieve(test["question"])
    ret_ms = (time.perf_counter() - r0) * 1000
    recall5 = recall10 = None
    if test.get("required_citations"):
        def hit_at(k):
            got = sum(1 for rc in test["required_citations"]
                      if any(e["filename"] == rc["filename"] and norm(rc["snippet"]) in norm(e["text"])
                             for e in ev[:k]))
            return got / len(test["required_citations"])
        recall5, recall10 = hit_at(5), hit_at(10)
    return {"id": test["id"], "category": test["category"], "question": test["question"],
            "expected": exp or test.get("allowed_statuses"), "actual_status": status,
            "citations": len(resp.get("citations", [])), "conflicts": len(resp.get("conflicts", [])),
            "citation_valid": valid, "citation_total": total,
            "recall_at_5": recall5, "recall_at_10": recall10,
            "passed": not failures, "failures": failures, "latency_ms": round(dt_ms, 1),
            "retrieve_ms": round(ret_ms, 1)}


def main_run():
    mode = (os.getenv("EVAL_MODE", "stubbed") if "--mode" not in sys.argv
            else sys.argv[sys.argv.index("--mode") + 1]).lower()
    assert mode in ("stubbed", "live"), "EVAL_MODE must be stubbed|live"
    manifest = json.loads(MANIFEST.read_text())
    tests = manifest["tests"]
    client = TestClient(main.app)

    # upload corpus once (realistic shared index)
    ocr_ok = True
    for fn in manifest["corpus"]:
        p = ROOT / fn
        with open(p, "rb") as f:
            r = client.post("/documents", files={"files": (Path(fn).name, f)})
        if r.status_code != 200:
            print("CORPUS upload failed for %s: %s" % (fn, r.text[:200]))
            if fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
                ocr_ok = False
    print("corpus: %d docs, %d chunks, ocr_engine=%s" % (len(main.DOCS), len(main.CHUNKS), main.OCR))

    if mode == "stubbed":
        q2t = {t["question"]: t for t in tests}
        main.llm_with_system = make_answer_stub(q2t)
    else:
        if not os.getenv("LLM_API_KEY"):
            print("EVAL_MODE=live requires LLM_API_KEY"); sys.exit(2)

    def sid_chunk_fn(question):
        ev, _, _ = main.retrieve(question)
        return {e["sid"]: e for e in ev}

    results, lat = [], []
    for t in tests:
        if t.get("needs_ocr") and not ocr_ok:
            results.append({"id": t["id"], "category": t["category"], "skipped": True,
                            "reason": "OCR engine unavailable", "latency_ms": 0})
            print("SKIP %-28s ocr unavailable" % t["id"])
            continue
        res = run_case(client, t, sid_chunk_fn)
        results.append(res)
        lat.append(res["latency_ms"])
        print("%s %-28s %-20s %s" % ("PASS" if res["passed"] else "FAIL", res["id"],
                                     res["actual_status"], "; ".join(res["failures"][:2])))

    ran = [r for r in results if not r.get("skipped")]
    passed = sum(1 for r in ran if r["passed"])
    metrics = compute_metrics(ran)
    latency = {"mean_ms": round(statistics.mean(lat), 1) if lat else 0,
               "median_ms": round(statistics.median(lat), 1) if lat else 0,
               "p95_ms": (round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1) if len(lat) >= 20 else None),
               "p95_note": None if len(lat) >= 20 else "too few samples (<20) for a meaningful p95"}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
              "stub_notes": STUB_NOTES if mode == "stubbed" else None,
              "summary": {"passed": passed, "failed": len(ran) - passed,
                          "skipped": len(results) - len(ran), "total": len(results)},
              "metrics": metrics, "latency": latency, "tests": results}
    REPORT.write_text(json.dumps(report, indent=2))
    print("-" * 60)
    print("EVALUATION MODE: %s | %d/%d passed, %d skipped" %
          (mode.upper(), passed, len(ran), len(results) - len(ran)))
    for k, v in metrics.items():
        print("  %-22s %s" % (k, "n/a" if v is None else round(v, 4) if isinstance(v, float) else v))
    print("report -> tests/report.json")
    return 0 if passed == len(ran) else 1


def compute_metrics(ran):
    m = {}
    tv = sum(r.get("citation_valid", 0) for r in ran)
    tt = sum(r.get("citation_total", 0) for r in ran)
    m["citation_validity"] = (tv / tt) if tt else None
    rec = [r["recall_at_5"] for r in ran if r.get("recall_at_5") is not None]
    m["citation_recall"] = (sum(rec) / len(rec)) if rec else None
    r5 = [r["recall_at_5"] for r in ran if r.get("recall_at_5") is not None]
    r10 = [r["recall_at_10"] for r in ran if r.get("recall_at_10") is not None]
    m["recall_at_5"] = (sum(r5) / len(r5)) if r5 else None
    m["recall_at_10"] = (sum(r10) / len(r10)) if r10 else None
    # conflict metrics: only single-status ground truth with/without expected conflict
    tp = fp = fn = tn = 0
    for r in ran:
        exp = r.get("expected")
        if isinstance(exp, list):
            continue  # ambiguous: excluded, no supported assumption
        want = (exp == "conflicting")
        got = (r["actual_status"] == "conflicting" and r["conflicts"] > 0)
        # a bare status without exposed pairs counts as missed/false
        if r["actual_status"] == "conflicting" and r["conflicts"] == 0:
            got = False
        tp += want and got
        fp += (not want) and (r["actual_status"] == "conflicting")
        fn += want and not got
        tn += (not want) and not (r["actual_status"] == "conflicting")
    m["conflict_precision"] = (tp / (tp + fp)) if (tp + fp) else None
    m["conflict_recall"] = (tp / (tp + fn)) if (tp + fn) else None
    m["conflict_f1"] = ((2 * m["conflict_precision"] * m["conflict_recall"] /
                         (m["conflict_precision"] + m["conflict_recall"]))
                        if m["conflict_precision"] and m["conflict_recall"] else
                        (0.0 if (tp + fp + fn) > 0 else None))
    # abstention: only single-status ground truth counts (ambiguous excluded)
    exp_abs = [r for r in ran if r.get("expected") == "insufficient_evidence"]
    got_abs = [r for r in ran if r.get("expected") in (None, "insufficient_evidence")
               and r["actual_status"] == "insufficient_evidence"]
    correct = [r for r in exp_abs if r["actual_status"] == "insufficient_evidence"]
    m["abstention_precision"] = (len(correct) / len(got_abs)) if got_abs else None
    m["abstention_recall"] = (len(correct) / len(exp_abs)) if exp_abs else None
    m["correct_abstentions"] = len(correct)
    m["false_confident_answers"] = sum(1 for r in exp_abs if r["actual_status"] != "insufficient_evidence")
    # unsupported-claim rate: answered tests lacking any valid citation
    answered = [r for r in ran if r["actual_status"] in ("supported", "conflicting")]
    bad = [r for r in answered if r.get("citation_valid", 0) == 0]
    m["unsupported_claim_rate"] = (len(bad) / len(answered)) if answered else None
    return m


if __name__ == "__main__":
    sys.exit(main_run())
