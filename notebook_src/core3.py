# =====================================================================
# SECTION 2b / 8b — EXECUTION MODES AND THE MODEL LAYER
# =====================================================================
# LIVE       real ChatAnthropic; needs ANTHROPIC_API_KEY
# REPLAY     AI turns recorded from a previous LIVE run, committed to disk
# SIMULATED  a deterministic stand-in so the notebook runs with no key.
#            It is NOT a language model and NOT a recording of one.
# =====================================================================
from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from core import (CLAIMS, CLAIMS_BY_ID, ENGINE, MATERIAL_INFO_CODES, REGISTRY,
                  Claim, money)
from core2 import (FAIL_INVALID_DRAFT, FAIL_LLM_ERROR, FAIL_NO_DRAFT,
                   AgentDraft, ClaimDecision, GateResult, RunAudit, ToolBelt,
                   claim_to_prompt_json, reconcile, run_agent, score_confidence,
                   build_packet, _engine_explanation)

MODEL_NAME = os.environ.get("REIMB_MODEL", "claude-sonnet-5")
TRANSCRIPT_PATH = Path("transcripts.json")

MODE_LIVE, MODE_REPLAY, MODE_SIMULATED = "LIVE", "REPLAY", "SIMULATED"


def detect_mode() -> str:
    forced = os.environ.get("REIMB_MODE", "").strip().upper()
    if forced in (MODE_LIVE, MODE_REPLAY, MODE_SIMULATED):
        return forced
    if os.environ.get("ANTHROPIC_API_KEY"):
        return MODE_LIVE
    if TRANSCRIPT_PATH.exists():
        return MODE_REPLAY
    return MODE_SIMULATED


MODE_BANNERS = {
    MODE_LIVE: ("LIVE  -  calling the Claude API. Tool selection and explanation prose are "
                "produced by the model."),
    MODE_REPLAY: ("REPLAY  -  no API key present. Replaying AI turns recorded from a previous "
                  "live run (see transcripts.json). Every number is recomputed now by the "
                  "deterministic engine."),
    MODE_SIMULATED: ("SIMULATED  -  no API key and no recorded transcript. A deterministic "
                     "stand-in drives the same LangChain AgentExecutor and the same tools. "
                     "It is NOT a language model and NOT a recording of one: tool selection "
                     "follows a fixed plan rather than being chosen turn by turn. Its "
                     "submission is built from the tool results it receives - it never reads "
                     "the engine - so a keyless run genuinely exercises the tool surface, the "
                     "precedence ladder and the reconciliation gate. What it does not "
                     "demonstrate is how a real model CHOOSES tools or words an explanation; "
                     "set ANTHROPIC_API_KEY for that."),
}


def _claim_id_from_messages(messages: Sequence[BaseMessage]) -> Optional[str]:
    for m in messages:
        if isinstance(m, HumanMessage):
            txt = m.content if isinstance(m.content, str) else json.dumps(m.content)
            s, e = txt.find("{"), txt.rfind("}")
            if s >= 0 and e > s:
                try:
                    return json.loads(txt[s:e + 1]).get("claim_id")
                except json.JSONDecodeError:
                    pass
            for cid in CLAIMS_BY_ID:
                if cid in txt:
                    return cid
    return None


def _tool_results(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            txt = m.content if isinstance(m.content, str) else json.dumps(m.content)
            try:
                out.append(json.loads(txt))
            except json.JSONDecodeError:
                out.append({"_raw": txt})
    return out


# Human-readable leads for the stand-in's own prose. Deliberately its own
# wording, not the engine's - a model writes its own sentences. Module scope,
# because pydantic turns a leading-underscore class attribute into a
# ModelPrivateAttr rather than leaving it a dict.
SIM_BLOCKER_LEAD = {
    "POLICY_EXCEPTION_AIRFARE_CLASS": "a non-economy fare (POL-AIR-01)",
    "AIRFARE_CLASS_UNVERIFIED": "a fare class the claim does not establish (POL-AIR-01)",
    "MISSING_REQUIRED_RECEIPT": "a missing itemized receipt (POL-RCT-01/02)",
    "EXCEEDS_AGENT_AUTHORITY": "a total above the agent's authority (POL-APR-03)",
    "LATE_SUBMISSION": "submission outside the 30-day window (POL-TIME-01)",
    "DATA_CONFLICT": "a contradiction between the line item and the trip dates",
    "POLICY_SILENT": "a scenario the policy does not cover",
    "UNKNOWN_CATEGORY": "an expense category the policy does not classify",
}


class SimulatedChatModel(BaseChatModel):
    """Deterministic stand-in for the LLM, so the notebook runs without a key.

    It drives the real AgentExecutor through the real tools and builds its
    submission from the tool results it receives - it never reads the engine.
    A keyless run therefore exercises the tool surface, the precedence ladder
    and the reconciliation gate for real, and a bug in any of them shows up as
    a disagreement at the gate rather than being hidden.

    What it is NOT: a language model. Its tool choices follow a fixed plan
    instead of being chosen turn by turn, and its prose is templated. Nothing
    here is evidence about how a real model selects tools or explains itself.
    """

    plan: List[str] = Field(default_factory=lambda: [
        "policy_lookup",
        "compute_limits",
        "check_receipt_completeness",
        "check_timeliness",
        "detect_conflicts",
        "check_approval_threshold",
        "submit_decision",
    ])

    @property
    def _llm_type(self) -> str:
        return "simulated-deterministic"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "SimulatedChatModel":
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        claim_id = _claim_id_from_messages(messages)
        claim = CLAIMS_BY_ID.get(claim_id) if claim_id else None
        results = _tool_results(messages)
        step = len(results)

        if step >= len(self.plan) or claim is None:
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="No further action."))])

        name = self.plan[step]
        if name == "policy_lookup":
            cats = sorted({l.category for l in claim.lines})
            args: Dict[str, Any] = {
                "query": ("eligibility, per-diem caps, receipt rules, approval thresholds and "
                          "submission window for: " + ", ".join(cats)),
                "categories": cats,
            }
        elif name == "check_approval_threshold":
            total = "0"
            for r in results:
                if isinstance(r, dict) and "reimbursable_total" in r:
                    total = r["reimbursable_total"]
            args = {"reimbursable_total": float(total)}
        elif name == "submit_decision":
            args = self._compose_submission(claim, results)
        else:
            args = {"claim_id": claim.claim_id}

        msg = AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"sim_{claim.claim_id}_{step}"}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @staticmethod
    def _find(results: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
        """Locate a tool payload by its distinctive key, the way a model would
        recognise which result is which - not by position in the plan."""
        return next((r for r in results
                     if isinstance(r, dict) and key in r), None)

    @classmethod
    def _compose_submission(cls, claim: Claim,
                            results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compose the submission FROM THE TOOL RESULTS.

        This function never touches the engine. Every figure, every finding and
        every rule id below is read out of the JSON the tools returned during
        this session, and the precedence ladder is applied to that JSON. That is
        what makes a SIMULATED run a real exercise of the tool surface: if a
        tool serialised something wrongly, or the ladder were applied wrongly,
        the draft would diverge from the engine and the reconciliation gate
        would record the disagreement instead of quietly agreeing with itself.
        """
        limits = cls._find(results, "trip_geometry")
        receipts = cls._find(results, "missing_docs")
        timing = cls._find(results, "days_to_submit")
        conflicts = cls._find(results, "arithmetic_reconciles")
        tier = cls._find(results, "tier_rule")

        if limits is None:
            # No arithmetic available: refuse to guess, escalate.
            return {
                "claim_id": claim.claim_id, "decision": "MANUAL_REVIEW",
                "approved_amount": 0.0, "deducted_amount": 0.0,
                "missing_docs": [], "policy_refs": [], "confidence": 0.0,
                "explanation": ("The limit-checking tool returned no result, so no amount "
                                "could be established and the claim is escalated rather "
                                "than guessed."),
            }

        lines = limits.get("lines", []) or []
        reimbursable = Decimal(str(limits.get("reimbursable_total", "0")))
        deducted = Decimal(str(limits.get("deducted_total", "0")))

        # ---- every blocker/conflict the tools reported, de-duplicated ----
        gathered: List[Dict[str, Any]] = []

        def soak(findings):
            for f in findings or []:
                if isinstance(f, dict) and f.get("severity") in ("blocker", "conflict"):
                    gathered.append(f)

        soak(limits.get("findings"))
        for ln in lines:
            soak(ln.get("findings"))
        if receipts:
            soak(receipts.get("findings"))
        if timing and timing.get("finding"):
            soak([timing["finding"]])
        if tier and tier.get("finding"):
            soak([tier["finding"]])
        if conflicts:
            soak(conflicts.get("conflicts"))

        seen, blockers = set(), []
        for f in gathered:
            key = (f.get("code"), f.get("line_id"), f.get("message"))
            if key not in seen:
                seen.add(key)
                blockers.append(f)

        # ---- precedence ladder, applied to the tool output ----
        all_ineligible = bool(lines) and all(l.get("eligible") is False for l in lines)
        if all_ineligible:
            decision = "REJECT"
        elif blockers:
            decision = "MANUAL_REVIEW"
        elif deducted > 0:
            decision = "PARTIAL_APPROVE"
        else:
            decision = "APPROVE"

        if decision == "MANUAL_REVIEW":
            approved, deducted_out = Decimal("0"), Decimal("0")
        elif decision == "REJECT":
            approved, deducted_out = Decimal("0"), deducted
        else:
            approved, deducted_out = reimbursable, deducted

        reason_codes = sorted({f.get("code") for f in blockers if f.get("code")})
        missing = list(receipts.get("missing_docs", [])) if (
            receipts and decision == "MANUAL_REVIEW") else []

        # ---- rules that drove this decision (same rule the engine applies) ----
        refs: List[str] = []

        def cite(rule_id):
            if rule_id and rule_id not in refs:
                refs.append(rule_id)

        def soak_refs(findings):
            for f in findings or []:
                if not isinstance(f, dict):
                    continue
                if f.get("severity") != "info" or f.get("code") in MATERIAL_INFO_CODES:
                    for r in f.get("policy_refs", []) or []:
                        cite(r)

        soak_refs(limits.get("findings"))
        for ln in lines:
            soak_refs(ln.get("findings"))
        if receipts:
            soak_refs(receipts.get("findings"))
        if timing and timing.get("finding"):
            soak_refs([timing["finding"]])
        if tier and tier.get("finding"):
            soak_refs([tier["finding"]])
        if conflicts:
            soak_refs(conflicts.get("conflicts"))
        if any(l.get("eligible") is True for l in lines):
            cite("POL-CAT-01")
        if any(l.get("eligible") is False for l in lines):
            cite("POL-CAT-02")
        tier_rule = (tier or {}).get("tier_rule")
        if tier and (decision in ("APPROVE", "PARTIAL_APPROVE")
                     or not tier.get("within_agent_authority", True)):
            cite(tier_rule)
        refs = sorted(refs)

        conf, _ = score_confidence(decision, reason_codes)
        return {
            "claim_id": claim.claim_id,
            "decision": decision,
            "approved_amount": float(approved),
            "deducted_amount": float(deducted_out),
            "missing_docs": missing,
            "policy_refs": refs,
            "confidence": conf,
            "explanation": cls._compose_explanation(
                decision, lines, reimbursable, deducted, blockers, reason_codes, tier_rule),
        }

    @classmethod
    def _compose_explanation(cls, decision, lines, reimbursable, deducted,
                             blockers, reason_codes, tier_rule) -> str:
        """Prose written from the tool output, within the 5-sentence limit the
        system prompt sets and using only figures the tools actually returned."""
        if decision == "APPROVE":
            return (f"All {len(lines)} line item(s) are eligible under POL-CAT-01, every "
                    f"required receipt is attached, and no amount exceeds its per-diem cap. "
                    f"The ${reimbursable} total is approvable under {tier_rule}.")
        if decision == "REJECT":
            return (f"Every claimed item is ineligible under POL-CAT-02, so nothing is "
                    f"reimbursable and ${deducted} is deducted in full. With no reimbursable "
                    f"content there is nothing for a reviewer to weigh, so the claim is "
                    f"rejected rather than escalated.")
        if decision == "PARTIAL_APPROVE":
            over = [l for l in lines if Decimal(str(l.get("excess", "0"))) > 0]
            bits = []
            for l in over[:2]:
                bits.append(f"{l['line_id']} claims ${l['claimed']} against a "
                            f"${l['cap_applied']} cap ({l['units_used']} "
                            f"{l['unit_kind']}(s)), so ${l['excess']} is deducted.")
            bits.append(f"The remaining ${reimbursable} is approvable under {tier_rule}.")
            return " ".join(bits)
        why = [SIM_BLOCKER_LEAD.get(c, c.lower().replace("_", " "))
               for c in reason_codes]
        joined = why[0] if len(why) == 1 else ", ".join(why[:-1]) + " and " + why[-1]
        return (f"This claim is escalated because of {joined}. "
                f"Reason code(s): {', '.join(reason_codes)}. "
                f"Provisionally ${reimbursable} would be reimbursable and ${deducted} "
                f"deducted, but no amount is authorised until a reviewer decides.")


class ReplayChatModel(BaseChatModel):
    """Replays AI turns captured from a previous LIVE run."""

    transcripts: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "replay"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ReplayChatModel":
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        claim_id = _claim_id_from_messages(messages) or ""
        turns = self.transcripts.get(claim_id, [])
        step = len(_tool_results(messages))
        if step >= len(turns):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="No further action."))])
        rec = turns[step]
        msg = AIMessage(content=rec.get("content", ""),
                        tool_calls=rec.get("tool_calls", []))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def make_llm(mode: str):
    if mode == MODE_LIVE:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=MODEL_NAME, temperature=0, max_tokens=2048,
                             timeout=90, stop=None)
    if mode == MODE_REPLAY:
        data = json.loads(TRANSCRIPT_PATH.read_text())
        return ReplayChatModel(transcripts=data.get("transcripts", data))
    return SimulatedChatModel()


# =====================================================================
# SECTION 10 — ORCHESTRATOR
# =====================================================================
class ClaimRun:
    """Everything produced for one claim: result, ledger, audit, gate trace."""

    def __init__(self, claim: Claim, decision: ClaimDecision, ledger, audit: RunAudit,
                 gate: GateResult):
        self.claim = claim
        self.decision = decision
        self.ledger = ledger
        self.audit = audit
        self.gate = gate

    @property
    def result(self) -> Dict[str, Any]:
        """The nine-field contract object."""
        return self.decision.model_dump()

    def audit_record(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim.claim_id,
            "mode": self.audit.mode,
            "engine_decision": self.gate.engine_decision,
            "llm_decision": self.gate.llm_decision,
            "disagreed": self.gate.disagreed,
            "gate_notes": self.gate.notes,
            "confidence_trace": self.gate.confidence_notes,
            "ledger": self.ledger.as_dict(),
            "agent": self.audit.as_dict(),
            "manual_review_packet": self.gate.packet.as_dict() if self.gate.packet else None,
            "final": self.result,
        }


def adjudicate_claim(claim: Claim, llm, mode: str) -> ClaimRun:
    """Pre-flight the engine, run the agent, then gate the result."""
    ledger = ENGINE.adjudicate(claim)                 # ground truth, held aside
    audit = RunAudit(claim_id=claim.claim_id, mode=mode)

    draft, _belt = run_agent(claim, llm, audit)

    # Classify by the typed failure kind, never by matching substrings of an
    # error message: an unreachable model falls back to the engine decision,
    # while an unusable answer is escalated at zero confidence.
    llm_down = audit.failure_kind == FAIL_LLM_ERROR
    unusable = audit.failure_kind in (FAIL_NO_DRAFT, FAIL_INVALID_DRAFT)

    gate = reconcile(claim, ledger, draft, audit, llm_unavailable=llm_down,
                     invalid_output=unusable)
    return ClaimRun(claim, gate.decision, ledger, audit, gate)


def adjudicate_all(claims: Optional[List[Claim]] = None, llm=None,
                   mode: Optional[str] = None) -> List[ClaimRun]:
    mode = mode or detect_mode()
    llm = llm if llm is not None else make_llm(mode)
    return [adjudicate_claim(c, llm, mode) for c in (claims or CLAIMS)]


def record_transcripts(path: Path = TRANSCRIPT_PATH) -> Dict[str, Any]:
    """Run LIVE and capture each claim's AI turns so the notebook can replay
    them later with no API key. Requires ANTHROPIC_API_KEY."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("record_transcripts() needs ANTHROPIC_API_KEY")

    from langchain_anthropic import ChatAnthropic

    captured: Dict[str, List[Dict[str, Any]]] = {}

    class Capturing(ChatAnthropic):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            cid = _claim_id_from_messages(messages) or "?"
            msg = res.generations[0].message
            captured.setdefault(cid, []).append({
                "content": msg.content if isinstance(msg.content, str) else "",
                "tool_calls": [
                    {"name": tc["name"], "args": tc["args"], "id": tc.get("id", "")}
                    for tc in (getattr(msg, "tool_calls", None) or [])
                ],
            })
            return res

    llm = Capturing(model=MODEL_NAME, temperature=0, max_tokens=2048, timeout=90, stop=None)
    runs = [adjudicate_claim(c, llm, MODE_LIVE) for c in CLAIMS]
    payload = {
        "recorded_model": MODEL_NAME,
        "claims": [c.claim_id for c in CLAIMS],
        "transcripts": captured,
    }
    path.write_text(json.dumps(payload, indent=2))
    return {"path": str(path), "claims": len(runs),
            "turns": {k: len(v) for k, v in captured.items()}}
