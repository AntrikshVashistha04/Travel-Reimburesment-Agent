# =====================================================================
# SECTION 13 — EVALUATION HARNESS
# =====================================================================
from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import ValidationError

from core import CLAIMS, CLAIMS_BY_ID, ENGINE, REGISTRY, Claim, LineItem, money
from core2 import (AgentDraft, ClaimDecision, RunAudit, reconcile)
from core3 import adjudicate_all, adjudicate_claim, detect_mode, make_llm

CONTRACT_FIELDS = {
    "claim_id", "decision", "approved_amount", "deducted_amount", "missing_docs",
    "policy_refs", "confidence", "explanation", "tools_used",
}

# Hand-derived from Appendix A before any code was written. See DESIGN.md §10.
GOLDEN: Dict[str, Tuple[str, Decimal, Decimal]] = {
    "CLM-001": ("APPROVE",         Decimal("1110.00"), Decimal("0.00")),
    "CLM-002": ("REJECT",          Decimal("0.00"),    Decimal("380.00")),
    "CLM-003": ("PARTIAL_APPROVE", Decimal("840.00"),  Decimal("100.00")),
    "CLM-004": ("MANUAL_REVIEW",   Decimal("0.00"),    Decimal("0.00")),
    "CLM-005": ("MANUAL_REVIEW",   Decimal("0.00"),    Decimal("0.00")),
}

ALL_TOOLS = {"policy_lookup", "compute_limits", "check_receipt_completeness",
             "check_approval_threshold", "check_timeliness", "detect_conflicts",
             "submit_decision"}


class Results:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(self, name: str, passed: bool, detail: str, bar: str = ""):
        self.rows.append({"test": name, "result": "PASS" if passed else "FAIL",
                          "pass_bar": bar, "detail": detail})

    @property
    def all_passed(self) -> bool:
        return all(r["result"] == "PASS" for r in self.rows)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)[["test", "result", "pass_bar", "detail"]]


def test_golden(runs) -> Tuple[bool, str, pd.DataFrame]:
    rows, ok = [], True
    for r in runs:
        d = r.result
        exp_dec, exp_a, exp_d = GOLDEN[d["claim_id"]]
        hit = (d["decision"] == exp_dec
               and money(d["approved_amount"]) == exp_a
               and money(d["deducted_amount"]) == exp_d)
        ok &= hit
        rows.append({
            "claim_id": d["claim_id"],
            "expected": f"{exp_dec} {exp_a}/{exp_d}",
            "actual": f"{d['decision']} {money(d['approved_amount'])}/"
                      f"{money(d['deducted_amount'])}",
            "match": "OK" if hit else "MISMATCH",
        })
    n = sum(1 for x in rows if x["match"] == "OK")
    return ok, f"{n}/{len(rows)} exact", pd.DataFrame(rows)


def test_schema(runs) -> Tuple[bool, str]:
    bad = []
    for r in runs:
        keys = set(r.result.keys())
        if keys != CONTRACT_FIELDS:
            bad.append((r.claim.claim_id, sorted(keys ^ CONTRACT_FIELDS)))
        try:
            ClaimDecision(**r.result)
        except ValidationError as e:
            bad.append((r.claim.claim_id, str(e)[:90]))
    return not bad, (f"{len(runs)}/{len(runs)} validate, exactly 9 fields"
                     if not bad else f"{bad}")


def test_grounding(runs) -> Tuple[bool, str]:
    bad = []
    for r in runs:
        for ref in r.result["policy_refs"]:
            if not REGISTRY.exists(ref):
                bad.append((r.claim.claim_id, ref))
    cited = {ref for r in runs for ref in r.result["policy_refs"]}
    return not bad, (f"0 hallucinated; {len(cited)}/{len(REGISTRY)} real rules cited"
                     if not bad else f"hallucinated: {bad}")


def test_engine_determinism(n: int = 20) -> Tuple[bool, str]:
    sigs = set()
    for _ in range(n):
        sigs.add(json.dumps([ENGINE.adjudicate(c).as_dict() for c in CLAIMS],
                            sort_keys=True))
    return len(sigs) == 1, f"{n} runs -> {len(sigs)} distinct ledger(s)"


def test_pipeline_stability(mode: str, n: int = 3) -> Tuple[bool, str]:
    sigs = set()
    llm = make_llm(mode)
    for _ in range(n):
        runs = adjudicate_all(llm=llm, mode=mode)
        sigs.add(json.dumps([(r.result["claim_id"], r.result["decision"],
                              r.result["approved_amount"], r.result["deducted_amount"])
                             for r in runs], sort_keys=True))
    return len(sigs) == 1, f"{n} full runs -> {len(sigs)} distinct outcome set(s)"


def test_enum_coverage(runs) -> Tuple[bool, str]:
    seen = {r.result["decision"] for r in runs}
    want = {"APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"}
    return seen == want, f"{len(seen)}/4 exercised: {sorted(seen)}"


def test_tool_exercise(runs) -> Tuple[bool, str]:
    used = {t for r in runs for t in r.result["tools_used"]}
    missing = ALL_TOOLS - used
    return not missing, (f"{len(used)}/{len(ALL_TOOLS)} tools exercised"
                         if not missing else f"never called: {sorted(missing)}")


# ---------------------------------------------------------------------
# Robustness: a deliberately corrupt claim. This is a TEST FIXTURE and is
# excluded from the five Appendix B results and from the dashboard.
# ---------------------------------------------------------------------
BROKEN_CLAIM = Claim(
    claim_id="FIXTURE-BAD",
    title="Robustness fixture (not part of Appendix B)",
    employee="Test Fixture",
    trip_start=date(2026, 6, 1),
    trip_end=date(2026, 6, 2),
    submitted_date=date(2026, 9, 30),          # 121 days late
    stated_total=Decimal("999.00"),            # does not match the lines
    lines=[
        LineItem(line_id="FIXTURE-BAD-L1", category="crypto_mining",
                 description="Unrecognised expense category",
                 amount=Decimal("500.00"), receipt_attached=False),
        LineItem(line_id="FIXTURE-BAD-L2", category="meals",
                 description="Meals, 9 days @ $200/day",
                 amount=Decimal("1800.00"), receipt_attached=False),
    ],
)


def test_robustness(mode: str) -> Tuple[bool, str]:
    try:
        run = adjudicate_claim(BROKEN_CLAIM, make_llm(mode), mode)
    except Exception as exc:                        # noqa: BLE001
        return False, f"pipeline raised {type(exc).__name__}: {exc}"
    d = run.result
    codes = run.gate.packet.reason_codes if run.gate.packet else []
    ok = d["decision"] == "MANUAL_REVIEW" and "UNKNOWN_CATEGORY" in codes
    return ok, f"{d['decision']} with {len(codes)} reason code(s): {', '.join(codes)}"


def test_input_validation() -> Tuple[bool, str]:
    """Pydantic must refuse structurally impossible claims."""
    caught = 0
    try:
        LineItem(line_id="X", category="meals", description="d",
                 amount=Decimal("-5"), receipt_attached=True)
    except ValidationError:
        caught += 1
    try:
        Claim(claim_id="X", title="t", employee="e",
              trip_start=date(2026, 6, 10), trip_end=date(2026, 6, 1),
              submitted_date=date(2026, 6, 20), stated_total=Decimal("1"), lines=[])
    except ValidationError:
        caught += 1
    return caught == 2, f"{caught}/2 malformed inputs rejected at the boundary"


# Expected elapsed days per claim, fixed by Appendix B's own dates. If any
# code ever reached for datetime.now(), today's date would leak in here and
# every claim would read as late - so this is the frozen-clock guard.
EXPECTED_ELAPSED = {"CLM-001": 10, "CLM-002": 11, "CLM-003": 14,
                    "CLM-004": 12, "CLM-005": 13}


def test_frozen_clock() -> Tuple[bool, str]:
    bad = []
    for c in CLAIMS:
        got = ENGINE.adjudicate(c).days_to_submit
        if got != EXPECTED_ELAPSED[c.claim_id]:
            bad.append((c.claim_id, got, EXPECTED_ELAPSED[c.claim_id]))
    return not bad, ("elapsed days computed from claim dates only, not today's date"
                     if not bad else f"wall-clock leak: {bad}")


def test_fallback_explanation_valid() -> Tuple[bool, str]:
    """The engine's own explanation is what the gate falls back TO, so it must
    pass the gate's own explanation check for every claim and every decision.
    If it ever fails, the fallback replaces itself and burns confidence."""
    from core2 import validate_explanation, _engine_explanation
    bad = []
    for c in CLAIMS + [BROKEN_CLAIM]:
        led = ENGINE.adjudicate(c)
        txt = _engine_explanation(c, led)
        ok, problems = validate_explanation(txt, led.candidate_decision, c, led)
        if not ok:
            bad.append((c.claim_id, led.candidate_decision, problems))
    return not bad, ("engine explanation passes the gate's own check on all "
                     f"{len(CLAIMS) + 1} claims" if not bad else f"{bad}")


def test_draft_is_tool_derived(mode: str) -> Tuple[bool, str]:
    """Prove the agent's draft follows the TOOLS rather than the engine.

    A doctored compute_limits payload is fed to the agent for CLM-003 - one
    that reports no per-diem excess. If the draft were derived from the engine
    it would be unmoved. It must instead follow the (wrong) tool output, the
    engine must hold its ground, and the gate must record the disagreement.
    This is what stops the golden table being the engine agreeing with itself.
    """
    import json as _json
    from core2 import ToolBelt

    original = ToolBelt.compute_limits

    def doctored(self, claim_id):
        out = _json.loads(original(self, claim_id))
        for ln in out["lines"]:
            ln["allowed"], ln["excess"] = ln["claimed"], "0.00"
            ln["findings"] = [f for f in ln.get("findings", [])
                              if f.get("code") != "OVER_PER_DIEM"]
        out["deducted_total"] = "0.00"
        out["reimbursable_total"] = "940.00"
        out["findings"] = []
        return _json.dumps(out)

    ToolBelt.compute_limits = doctored
    try:
        run = adjudicate_claim(CLAIMS_BY_ID["CLM-003"], make_llm(mode), mode)
    finally:
        ToolBelt.compute_limits = original

    drafted = (run.audit.draft or {}).get("decision")
    shipped = run.result["decision"]
    ok = (drafted == "APPROVE" and shipped == "PARTIAL_APPROVE" and run.gate.disagreed)
    return ok, (f"doctored tool output -> agent drafted {drafted}, engine held {shipped}, "
                f"gate flagged disagreement={run.gate.disagreed}")


def test_agent_engine_agreement(runs) -> Tuple[bool, str]:
    """With honest tools, the independently-derived draft should match the
    engine on every claim. Now that the draft comes from tool output rather
    than from the ledger, this agreement is evidence rather than a tautology."""
    disagreed = [r.claim.claim_id for r in runs if r.gate.disagreed]
    # A draft that never arrived is not agreement - an agent that crashed would
    # otherwise sail through this check while the engine quietly carried it.
    nodraft = [r.claim.claim_id for r in runs if not r.audit.draft]
    failed = [f"{r.claim.claim_id}:{r.audit.failure_kind}" for r in runs
              if r.audit.failure_kind]
    ok = not disagreed and not nodraft and not failed
    if ok:
        return True, (f"{len(runs)}/{len(runs)} drafts produced and matched the engine "
                      f"independently, with no agent failures")
    return False, (f"disagreed={disagreed or 'none'} no_draft={nodraft or 'none'} "
                   f"failures={failed or 'none'}")


def run_evaluation(runs, mode: str) -> Results:
    res = Results()
    ok, detail, golden_df = test_golden(runs)
    res.add("Golden outcomes", ok, detail, "5/5 exact")
    for name, bar, fn in (
        ("Schema conformance", "5/5 valid, 9 fields", lambda: test_schema(runs)),
        ("Policy-ref grounding", "0 hallucinated", lambda: test_grounding(runs)),
        ("Engine determinism", "identical x20", test_engine_determinism),
        ("Pipeline stability", "identical x3", lambda: test_pipeline_stability(mode)),
        ("Decision enum coverage", "4/4", lambda: test_enum_coverage(runs)),
        ("Tool exercise", "7/7 called", lambda: test_tool_exercise(runs)),
        ("Robustness fixture", "escalates, no crash", lambda: test_robustness(mode)),
        ("Input validation", "2/2 rejected", test_input_validation),
        ("Frozen clock", "no wall-clock leak", test_frozen_clock),
        ("Fallback explanation", "valid on all claims", test_fallback_explanation_valid),
        ("Draft is tool-derived", "follows doctored tools",
         lambda: test_draft_is_tool_derived(mode)),
        ("Agent/engine agreement", "5/5 independently",
         lambda: test_agent_engine_agreement(runs)),
    ):
        passed, detail = fn()
        res.add(name, passed, detail, bar)
    res.golden_df = golden_df                      # type: ignore[attr-defined]
    return res


# =====================================================================
# SECTION 14 — ABLATION: WHAT THE RECONCILIATION GATE IS FOR
# =====================================================================
# Runs with no API key. A deliberately wrong agent draft is pushed through
# the gate to show, concretely, what an ungrounded model would have cost.
# =====================================================================
ADVERSARIAL_DRAFT = AgentDraft(
    claim_id="CLM-004",
    decision="APPROVE",                 # wrong: three independent blockers apply
    approved_amount=3000.00,            # wrong: nothing is authorised
    deducted_amount=0.00,               # wrong: $200 over the lodging cap
    missing_docs=[],                    # wrong: the lodging receipt is absent
    policy_refs=["POL-AIR-01", "POL-FAKE-99", "POL-APR-99"],   # two are invented
    confidence=0.99,                    # wrong: overconfident
    explanation="Business-class travel was pre-approved for this vendor negotiation, "
                "so the full amount is reimbursable and no receipt is required.",
)


def run_gate_ablation() -> Dict[str, Any]:
    claim = CLAIMS_BY_ID["CLM-004"]
    ledger = ENGINE.adjudicate(claim)
    audit = RunAudit(claim_id=claim.claim_id, mode="ABLATION")
    audit.calls = []
    gate = reconcile(claim, ledger, ADVERSARIAL_DRAFT, audit)

    ungated = {
        "decision": ADVERSARIAL_DRAFT.decision,
        "approved_amount": ADVERSARIAL_DRAFT.approved_amount,
        "deducted_amount": ADVERSARIAL_DRAFT.deducted_amount,
        "missing_docs": ADVERSARIAL_DRAFT.missing_docs,
        "policy_refs": ADVERSARIAL_DRAFT.policy_refs,
        "confidence": ADVERSARIAL_DRAFT.confidence,
    }
    gated = {
        "decision": gate.decision.decision,
        "approved_amount": gate.decision.approved_amount,
        "deducted_amount": gate.decision.deducted_amount,
        "missing_docs": gate.decision.missing_docs,
        "policy_refs": gate.decision.policy_refs,
        "confidence": gate.decision.confidence,
    }
    comparison = pd.DataFrame([
        {"field": k,
         "ungated (model trusted)": str(ungated[k]),
         "gated (shipped)": str(gated[k]),
         "changed": "yes" if str(ungated[k]) != str(gated[k]) else "no"}
        for k in ungated
    ])
    return {
        "comparison": comparison,
        "gate_notes": gate.notes,
        "money_at_risk": ADVERSARIAL_DRAFT.approved_amount - gate.decision.approved_amount,
        "fake_refs_dropped": [r for r in ADVERSARIAL_DRAFT.policy_refs
                              if not REGISTRY.exists(r)],
    }


def run_llm_only_ablation(mode: str) -> Dict[str, Any]:
    """The true LLM-vs-pipeline ablation: ask the model to adjudicate with no
    tools, no policy and no engine, then diff against the shipped result.
    Only meaningful against a real model, so it is skipped off LIVE."""
    if mode != "LIVE":
        return {"skipped": True,
                "reason": f"Requires a live model; current mode is {mode}. "
                          f"Set ANTHROPIC_API_KEY and re-run to execute it."}
    from langchain_anthropic import ChatAnthropic
    from core2 import claim_to_prompt_json
    from core3 import MODEL_NAME

    llm = ChatAnthropic(model=MODEL_NAME, temperature=0, max_tokens=1024,
                        timeout=90, stop=None)
    rows = []
    for claim in CLAIMS:
        msg = llm.invoke(
            "You are a travel reimbursement adjudicator. Decide this claim and reply "
            "ONLY with JSON {\"decision\":..., \"approved_amount\":..., "
            "\"deducted_amount\":...}. decision is one of APPROVE, PARTIAL_APPROVE, "
            "REJECT, MANUAL_REVIEW.\n\n" + claim_to_prompt_json(claim))
        txt = msg.content if isinstance(msg.content, str) else str(msg.content)
        s, e = txt.find("{"), txt.rfind("}")
        try:
            got = json.loads(txt[s:e + 1])
        except Exception:                            # noqa: BLE001
            got = {"decision": "?", "approved_amount": None, "deducted_amount": None}
        exp_dec, exp_a, exp_d = GOLDEN[claim.claim_id]
        rows.append({
            "claim_id": claim.claim_id,
            "llm_only": f"{got.get('decision')} {got.get('approved_amount')}/"
                        f"{got.get('deducted_amount')}",
            "correct": f"{exp_dec} {exp_a}/{exp_d}",
            "match": "OK" if (got.get("decision") == exp_dec
                              and money(got.get("approved_amount") or 0) == exp_a
                              and money(got.get("deducted_amount") or 0) == exp_d)
                     else "WRONG",
        })
    df = pd.DataFrame(rows)
    return {"skipped": False, "comparison": df,
            "accuracy": f"{(df['match'] == 'OK').sum()}/{len(df)}"}
