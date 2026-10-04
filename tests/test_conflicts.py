"""Focused Phase 2 unit tests: normalization, pairing, qualifier handling, FP prevention."""
import conflicts as C


def ev(sid, text, doc="d1", fn="doc.txt"):
    return {"sid": sid, "doc_id": doc, "filename": fn, "page": 1,
            "section": "", "method": "text", "text": text}


# ---- date normalization ----
def test_dates_equivalent():
    assert C.normalize_date("30 June 2027") == "2027-06-30"
    assert C.normalize_date("June 30, 2027") == "2027-06-30"
    assert C.normalize_date("30/06/2027") == "2027-06-30"
    assert C.normalize_date("2027-06-30") == "2027-06-30"
    assert C.normalize_date("15th March 2027") == "2027-03-15"


def test_dates_ambiguous_left_alone():
    assert C.normalize_date("03/04/2027") is None       # both <= 12: refuse
    assert C.normalize_date("13/04/2027") == "2027-04-13"  # day-first, unambiguous
    assert C.normalize_date("04/13/2027") == "2027-04-13"  # month-first, unambiguous
    assert C.normalize_date("30 June") is None           # no year
    assert C.normalize_date("not a date") is None


# ---- amount normalization ----
def test_amounts_inr_forms():
    assert C.normalize_amount("INR 12,00,000") == ("INR", 1200000.0)
    assert C.normalize_amount("INR 12 lakh") == ("INR", 1200000.0)
    assert C.normalize_amount("12 lakhs") == (None, 1200000.0)
    assert C.normalize_amount("Rs. 14.5 lakh") == ("INR", 1450000.0)
    assert C.normalize_amount("INR 14,50,000") == ("INR", 1450000.0)


def test_amounts_currency_safety():
    assert C.normalize_amount("USD 10,000") == ("USD", 10000.0)
    assert C.normalize_amount("INR 10,000") == ("INR", 10000.0)
    # different currencies must never compare equal downstream (checked in pairing test)
    assert C.normalize_amount("10 blorps") is None
    assert C.normalize_amount("much money") is None


# ---- percentage / version / quantity ----
def test_percentage_forms():
    assert C.normalize_percentage("10%") == 10.0
    assert C.normalize_percentage("10 percent") == 10.0
    assert C.normalize_percentage("ten percent") is None


def test_version_forms():
    assert C.normalize_version("v2") == C.normalize_version("Version 2") == C.normalize_version("v2.0") == (2,)
    assert C.normalize_version("v2.1") == (2, 1)
    assert C.normalize_version("banana") is None


def test_quantity_forms():
    assert C.normalize_quantity("100 units") == (100.0, "units")
    assert C.normalize_quantity("100") is None  # bare numbers too noisy


# ---- pairing: spec section 13 cases ----
def claims_for(texts):
    out = []
    for i, t in enumerate(texts):
        out.extend(C.extract_claims([ev("S%d" % (i + 1), t, doc="d%d" % (i + 1))]))
    return out


def test_1_date_conflict_candidate():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    assert len(C.pair_candidates(cs)) == 1


def test_2_qualifier_case_goes_to_verifier():
    cs = claims_for(["Project Atlas forecast budget is INR 12 lakh.",
                     "Project Atlas approved budget is INR 14.5 lakh."])
    cands = C.pair_candidates(cs)
    assert len(cands) == 1  # candidate, NOT auto-confirmed
    assert sorted(cands[0]["claim_a"]["qualifiers"]) == ["forecast"]


def test_3_equivalent_dates_compatible():
    cs = claims_for(["Project Atlas release date is 2027-06-30.",
                     "Project Atlas release date is 30 June 2027."])
    assert C.pair_candidates(cs) == []


def test_4_percent_compatible():
    cs = claims_for(["Project Atlas growth is 10%.",
                     "Project Atlas growth is 10 percent."])
    assert C.pair_candidates(cs) == []


def test_5_version_compatible():
    cs = claims_for(["Project Atlas version is Version 2.",
                     "Project Atlas version is v2.0."])
    assert C.pair_candidates(cs) == []


def test_6_different_subjects_no_pair():
    cs = claims_for(["Project A release date: 30 June 2027.",
                     "Project B release date: 15 July 2027."])
    assert C.pair_candidates(cs) == []


def test_7_different_periods_no_pair():
    cs = claims_for(["Project Atlas 2026 revenue was INR 10 lakh.",
                     "Project Atlas 2027 revenue is INR 15 lakh."])
    assert C.pair_candidates(cs) == []


def test_8_draft_revised_is_candidate():
    cs = claims_for(["Project Atlas draft schedule: 30 June 2027.",
                     "Project Atlas revised schedule: 15 July 2027."])
    assert len(C.pair_candidates(cs)) == 1


def test_currency_mismatch_no_pair():
    cs = claims_for(["Project Atlas budget is USD 10,000.",
                     "Project Atlas budget is INR 10,000."])
    assert C.pair_candidates(cs) == []


def test_same_passage_no_self_pair():
    cs = C.extract_claims([ev("S1", "Project Atlas release date is 30 June 2027. "
                                     "Project Atlas launch date is 15 July 2027.")])
    assert C.pair_candidates(cs) == []


def test_subject_inherited_within_document():
    cs = C.extract_claims([
        ev("S1", "Revised Delivery Timeline. The revised production release date is 15 July 2027.",
           doc="doc3", fn="3_revised_schedule.txt"),
        ev("S2", "Revised Budget. The revised budget for Project Atlas is INR 14,50,000.",
           doc="doc3", fn="3_revised_schedule.txt"),
        ev("S3", "Delivery Timeline. The production release will be completed on 30 June 2027. "
                 "Project Atlas pilot is fine.", doc="doc1", fn="1_vendor_proposal.txt"),
    ])
    by_subj = {}
    for cl in cs:
        by_subj.setdefault(cl["citation_id"], cl["subject"])
    assert by_subj["S1"] == "Project Atlas"  # inherited from sibling passage S2
    dates = [cl for cl in cs if cl["value_type"] == "date" and cl["attribute"] == "release date"]
    assert len(C.pair_candidates(dates)) >= 1


def test_relevance_filter_keeps_question_pairs():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027.",
                     "Project Atlas budget is INR 12 lakh.",
                     "Project Atlas budget is INR 14 lakh."])
    cands = C.pair_candidates(cs)
    assert len(cands) == 2
    date_q = C.filter_question_relevant(cands, "What is the production release date?")
    assert len(date_q) == 1 and date_q[0]["claim_a"]["attribute"] == "release date"
    budget_q = C.filter_question_relevant(cands, "What is the budget?")
    assert len(budget_q) == 1 and budget_q[0]["claim_a"]["attribute"] == "budget"
    assert C.filter_question_relevant(cands, "Who owns the trademark?") == []


# ---- verifier behavior (stubbed LLM) ----
def test_verifier_compatible_drops_candidate():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    cands = C.pair_candidates(cs)
    out = C.verify_candidates(cands, lambda s, u: {"classification": "compatible", "reason": "ok"})
    assert out == []


def test_verifier_failure_fails_open():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    cands = C.pair_candidates(cs)
    def boom(s, u):
        raise RuntimeError("no network")
    out = C.verify_candidates(cands, boom)
    assert len(out) == 1 and out[0]["resolution"] == "unresolved"


def test_verifier_bad_shape_fails_open():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    out = C.verify_candidates(C.pair_candidates(cs), lambda s, u: {"nope": 1})
    assert len(out) == 1


def test_hints_only_from_explicit_signals():
    by = {"S1": {"text": "Original proposal. Release 30 June 2027."},
          "S2": {"text": "This document supersedes the proposal. Revised date 15 July 2027."}}
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    cands = C.pair_candidates(cs)
    out = C.verify_candidates(cands, lambda s, u: {"classification": "conflict", "reason": "dates differ"})
    C.attach_hints(out, by)
    assert out[0]["likely_newer"] == "S2" and "hint only" in out[0]["newer_reason"]
    out2 = C.verify_candidates(cands, lambda s, u: {"classification": "conflict", "reason": "x"})
    C.attach_hints(out2, {"S1": {"text": "plain"}, "S2": {"text": "plain"}})
    assert out2[0]["likely_newer"] is None


def test_doc_subjects_span_unretrieved_chunks():
    chunks = [{"doc_id": "d3", "text": "Revised Delivery Timeline. Nothing else."},
              {"doc_id": "d3", "text": "The budget for Project Atlas is INR 5 lakh."}]
    subs = C.doc_subjects_for(chunks)
    assert subs == {"d3": "Project Atlas"}
    cs = C.extract_claims([ev("S1", "The revised production release date is 15 July 2027.", doc="d3")], subs)
    assert cs and cs[0]["subject"] == "Project Atlas"


def test_relevance_requires_specific_token():
    cs = claims_for(["Project Atlas release date is 30 June 2027.",
                     "Project Atlas release date is 15 July 2027."])
    cands = C.pair_candidates(cs)
    assert len(cands) == 1
    assert C.filter_question_relevant(cands, "What is the pilot date?") == []

