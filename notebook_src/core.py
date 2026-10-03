# =====================================================================
# SECTION 3 — APPENDIX A AS CODE: THE POLICY REGISTRY
# =====================================================================
# Every rule below is transcribed VERBATIM from Appendix A of the brief.
# `text` is the policy's own wording; `params` is the machine-readable
# projection the engine computes with. Nothing here is invented: if a
# number appears in params, it appears in the text beside it.
# =====================================================================
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

CURRENCY = "USD"
MONEY = Decimal("0.01")


def money(value) -> Decimal:
    """Quantise to cents. All financial arithmetic flows through Decimal."""
    return Decimal(str(value)).quantize(MONEY, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    title: str
    text: str                       # verbatim Appendix A wording
    rule_type: str                  # eligibility | limit | receipt | threshold | timeliness
    params: Dict[str, Any] = field(default_factory=dict)
    keywords: Tuple[str, ...] = ()

    def as_context(self) -> Dict[str, Any]:
        """Shape handed to the LLM when this rule is retrieved."""
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "text": self.text,
            "rule_type": self.rule_type,
            "params": {k: str(v) for k, v in self.params.items()},
        }


# --- 1. Eligible & Ineligible Categories -----------------------------
ELIGIBLE_CATEGORIES = (
    "airfare",
    "lodging",
    "meals",
    "ground_transport",
    "conference_fees",
)

# Appendix A POL-CAT-02 bullet list, normalised to claim-category tokens.
INELIGIBLE_CATEGORIES = (
    "alcohol",
    "minibar",
    "spa",
    "gym",
    "personal_entertainment",
    "in_room_movies",
    "personal_shopping",
    "gifts",
    "traffic_fines",
    "penalties",
    "late_fees",
    "personal",
)

POLICY_RULES: Tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="POL-CAT-01",
        title="Eligible categories",
        text=(
            "Reimbursable when incurred for a documented business purpose: "
            "Airfare (economy class only - see POL-AIR-01); Lodging (hotel room charges); "
            "Meals (subject to per-diem limits - see POL-PD-01); Ground transport "
            "(taxi, rideshare, train, rental car, parking); Conference / registration fees."
        ),
        rule_type="eligibility",
        params={"eligible_categories": list(ELIGIBLE_CATEGORIES)},
        keywords=("eligible", "category", "airfare", "lodging", "meals",
                  "ground", "transport", "conference", "registration", "business purpose"),
    ),
    PolicyRule(
        rule_id="POL-CAT-02",
        title="Ineligible items",
        text=(
            "Never reimbursable; rejected (deducted in full): Alcohol and minibar charges; "
            "Spa, gym, and personal entertainment; In-room movies, personal shopping, gifts; "
            "Traffic fines, penalties, and late fees; Any personal (non-business) expense."
        ),
        rule_type="eligibility",
        params={"ineligible_categories": list(INELIGIBLE_CATEGORIES)},
        keywords=("ineligible", "alcohol", "minibar", "spa", "gym", "entertainment",
                  "movies", "shopping", "gifts", "fines", "penalties", "late fees",
                  "personal", "rejected"),
    ),
    # --- 2. Per-Diem & Category Limits -------------------------------
    PolicyRule(
        rule_id="POL-PD-01",
        title="Meals per-diem",
        text=("Meals - Maximum $75 per day. Amounts above the daily cap are deducted; "
              "the rest is reimbursed."),
        rule_type="limit",
        params={"category": "meals", "cap": Decimal("75"), "unit": "day"},
        keywords=("meals", "per-diem", "per diem", "daily", "cap", "food", "dinner", "75"),
    ),
    PolicyRule(
        rule_id="POL-PD-02",
        title="Lodging per-night cap",
        text=("Lodging - Maximum $200 per night. Amounts above the nightly cap are deducted; "
              "the rest is reimbursed."),
        rule_type="limit",
        params={"category": "lodging", "cap": Decimal("200"), "unit": "night"},
        keywords=("lodging", "hotel", "night", "nightly", "cap", "room", "200"),
    ),
    PolicyRule(
        rule_id="POL-PD-03",
        title="Ground transport cap",
        text="Ground transport - Maximum $50 per day. Amounts above the cap are deducted.",
        rule_type="limit",
        params={"category": "ground_transport", "cap": Decimal("50"), "unit": "day"},
        keywords=("ground", "transport", "taxi", "rideshare", "train", "rental",
                  "parking", "daily", "50"),
    ),
    PolicyRule(
        rule_id="POL-AIR-01",
        title="Airfare class",
        text=("Only economy class airfare is reimbursable. Business/first-class fares are a "
              "policy exception and must be routed to Manual Review (not auto-deducted, "
              "because a pre-approval may exist)."),
        rule_type="limit",
        params={
            "category": "airfare",
            "allowed_class": "economy",
            "exception_route": "MANUAL_REVIEW",
            "auto_deduct": False,
        },
        keywords=("airfare", "flight", "economy", "business class", "business-class",
                  "first class", "first-class", "class", "exception", "pre-approval"),
    ),
    # --- 3. Receipt Rules --------------------------------------------
    PolicyRule(
        rule_id="POL-RCT-01",
        title="Receipt required above $25",
        text=("Any single line item greater than $25 requires an attached, itemized receipt. "
              "Airfare and lodging always require a receipt regardless of amount."),
        rule_type="receipt",
        params={
            "threshold": Decimal("25"),
            "always_required_categories": ["airfare", "lodging"],
        },
        keywords=("receipt", "itemized", "itemised", "attached", "25", "documentation",
                  "airfare", "lodging"),
    ),
    PolicyRule(
        rule_id="POL-RCT-02",
        title="Missing receipt handling",
        text=("If a receipt is missing for an item that requires one, the item is not silently "
              "rejected - the claim is routed to Manual Review so the reviewer can request "
              "the receipt."),
        rule_type="receipt",
        params={"route": "MANUAL_REVIEW", "silent_reject": False},
        keywords=("missing", "receipt", "manual review", "request", "reviewer"),
    ),
    # --- 4. Approval Thresholds --------------------------------------
    # Preamble, verbatim: "Thresholds are evaluated on the total reimbursable
    # amount (after per-diem deductions but before final decision)."
    PolicyRule(
        rule_id="POL-APR-01",
        title="Auto-approve tier",
        text="Total <= $500: may be auto-approved by the agent if fully compliant.",
        rule_type="threshold",
        params={"upper": Decimal("500"), "tier": "auto_approve", "agent_authority": True},
        keywords=("threshold", "auto-approve", "auto approve", "tier", "500", "approval"),
    ),
    PolicyRule(
        rule_id="POL-APR-02",
        title="Manager tier",
        text=("Total > $500 and <= $2,000: eligible for approval, treated as approvable when "
              "fully compliant."),
        rule_type="threshold",
        params={
            "lower": Decimal("500"),
            "upper": Decimal("2000"),
            "tier": "manager",
            "agent_authority": True,
        },
        keywords=("threshold", "manager", "tier", "2000", "2,000", "approval", "approvable"),
    ),
    PolicyRule(
        rule_id="POL-APR-03",
        title="Director / Manual-Review tier",
        text=("Total > $2,000: exceeds the agent's auto-approval authority and must be routed "
              "to Manual Review (director approval required), even if otherwise compliant."),
        rule_type="threshold",
        params={
            "lower": Decimal("2000"),
            "tier": "director",
            "agent_authority": False,
            "route": "MANUAL_REVIEW",
        },
        keywords=("threshold", "director", "high value", "2000", "2,000", "authority",
                  "manual review"),
    ),
    # --- 5. Timeliness -----------------------------------------------
    PolicyRule(
        rule_id="POL-TIME-01",
        title="Submission window",
        text=("Claims must be submitted within 30 days of the expense date. Late claims are "
              "routed to Manual Review."),
        rule_type="timeliness",
        params={"window_days": 30, "route": "MANUAL_REVIEW"},
        keywords=("timeliness", "submission", "window", "30 days", "late", "deadline"),
    ),
)

# Appendix A's closing "Decision Guidance" block, verbatim. It carries no
# POL-* id of its own, so it is held separately and never emitted as a
# policy_ref - that would mean inventing an identifier.
DECISION_GUIDANCE = (
    "Approve - every item eligible, all receipts present, all within per-diem, total within an "
    "approvable tier (POL-APR-01 / POL-APR-02).\n"
    "Partially Approve - the claim is valid but some amounts exceed per-diem caps; reimburse up "
    "to the cap and deduct the excess.\n"
    "Reject - the claimed items are ineligible (POL-CAT-02) with nothing reimbursable.\n"
    "Manual Review - any ambiguity, policy exception, high value (POL-APR-03), missing required "
    "receipt (POL-RCT-02), or conflicting information. Prefer Manual Review over forcing "
    "a decision."
)

THRESHOLD_BASIS = (
    "Thresholds are evaluated on the total reimbursable amount (after per-diem deductions but "
    "before final decision)."
)


class PolicyRegistry:
    """Typed, exact lookup over Appendix A. Deliberately not a vector store."""

    def __init__(self, rules: Tuple[PolicyRule, ...] = POLICY_RULES):
        self._rules = rules
        self._by_id = {r.rule_id: r for r in rules}

    def __len__(self) -> int:
        return len(self._rules)

    @property
    def ids(self) -> List[str]:
        return [r.rule_id for r in self._rules]

    def get(self, rule_id: str) -> Optional[PolicyRule]:
        return self._by_id.get(rule_id.strip().upper())

    def exists(self, rule_id: str) -> bool:
        return rule_id.strip().upper() in self._by_id

    def by_category(self, category: str) -> List[PolicyRule]:
        cat = category.strip().lower()
        hits = []
        for r in self._rules:
            if r.params.get("category") == cat:
                hits.append(r)
            elif cat in r.params.get("eligible_categories", []):
                hits.append(r)
            elif cat in r.params.get("ineligible_categories", []):
                hits.append(r)
            elif cat in r.params.get("always_required_categories", []):
                hits.append(r)
        return hits

    def search(self, query: str) -> List[PolicyRule]:
        """Keyword/substring scoring. Exact and auditable; no embeddings."""
        q = query.strip().lower()
        if not q:
            return []
        terms = [t for t in re.split(r"[^a-z0-9$,\-]+", q) if t]
        scored = []
        for r in self._rules:
            score = 0
            if r.rule_id.lower() in q:
                score += 100
            for kw in r.keywords:
                if kw in q:
                    score += 10
                elif any(t == kw for t in terms):
                    score += 8
            for t in terms:
                if len(t) > 3 and t in r.text.lower():
                    score += 2
                if len(t) > 3 and t in r.title.lower():
                    score += 3
            if score:
                scored.append((score, r))
        scored.sort(key=lambda p: (-p[0], p[1].rule_id))
        return [r for _, r in scored]


REGISTRY = PolicyRegistry()


# =====================================================================
# SECTION 4 — APPENDIX B AS DATA: THE FIVE CLAIMS
# =====================================================================
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Signals the engine extracts from each line item's own description text.
# Appendix B states quantities inside the description ("Hotel, 2 nights @ $180",
# "Client dinner for 4"), so they are PARSED from the supplied string rather
# than hand-entered - the extraction is part of the demonstrable pipeline.
_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(night|day)s?\b", re.I)
_RATE_RE = re.compile(r"@\s*~?\s*\$?\s*(\d+(?:\.\d+)?)", re.I)

# Headcount. "for N" alone is ambiguous: "Hotel for 3 nights" and "Taxi for 2
# trips" are quantities, not people. Only an explicit people-noun, or a bare
# "for N" NOT followed by a unit noun, counts as a headcount.
_HEAD_UNIT_NOUNS = (r"night|day|week|weekend|month|year|hour|trip|leg|flight|"
                    r"mile|km|kilometre|kilometer|ride|journey|stay|booking")
_HEAD_EXPLICIT_RE = re.compile(
    r"\bfor\s+(\d+)\s+(?:people|persons?|guests?|attendees?|diners?|pax|"
    r"colleagues?|clients?|staff)\b", re.I)
_HEAD_BARE_RE = re.compile(
    r"\bfor\s+(\d+)\b(?!\s*(?:%s)s?\b)" % _HEAD_UNIT_NOUNS, re.I)

# Fare class. POL-AIR-01 reimburses economy only and routes business/first to
# Manual Review. "Premium economy" is neither, and an unstated class cannot be
# verified at all - both are ambiguity, which the Decision Guidance sends to
# Manual Review rather than letting it default to economy.
_PREMIUM_ECONOMY_RE = re.compile(r"\bpremium[\s\-]*economy\b", re.I)
_PREMIUM_RE = re.compile(r"\b(business|first)[\s\-]*class\b", re.I)
_ECONOMY_RE = re.compile(r"\beconomy\b", re.I)


@dataclass(frozen=True)
class LineSignals:
    stated_units: Optional[Decimal] = None
    stated_unit_kind: Optional[str] = None     # "night" | "day"
    stated_rate: Optional[Decimal] = None
    headcount: Optional[int] = None
    # "economy" | "premium" | "premium_economy" | None (not stated)
    fare_class: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stated_units": None if self.stated_units is None else str(self.stated_units),
            "stated_unit_kind": self.stated_unit_kind,
            "stated_rate": None if self.stated_rate is None else str(self.stated_rate),
            "headcount": self.headcount,
            "fare_class": self.fare_class,
        }


def parse_line_signals(description: str) -> LineSignals:
    """Extract structured signals from an Appendix B description string."""
    units = kind = rate = head = fare = None
    m = _UNIT_RE.search(description)
    if m:
        units = Decimal(m.group(1))
        kind = m.group(2).lower()
    m = _RATE_RE.search(description)
    if m:
        rate = Decimal(m.group(1))
    m = _HEAD_EXPLICIT_RE.search(description) or _HEAD_BARE_RE.search(description)
    if m:
        head = int(m.group(1))
    if _PREMIUM_ECONOMY_RE.search(description):
        fare = "premium_economy"
    elif _PREMIUM_RE.search(description):
        fare = "premium"
    elif _ECONOMY_RE.search(description):
        fare = "economy"
    return LineSignals(units, kind, rate, head, fare)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    line_id: str
    category: str
    description: str
    amount: Decimal
    receipt_attached: bool

    @field_validator("category")
    @classmethod
    def _norm_category(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("amount")
    @classmethod
    def _non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("line amount may not be negative")
        return money(v)

    @property
    def signals(self) -> LineSignals:
        return parse_line_signals(self.description)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    title: str
    employee: str
    trip_start: date
    trip_end: date
    submitted_date: date
    stated_total: Decimal
    lines: List[LineItem]

    @model_validator(mode="after")
    def _dates_ordered(self) -> "Claim":
        if self.trip_end < self.trip_start:
            raise ValueError(f"{self.claim_id}: trip_end precedes trip_start")
        return self

    # Derived trip geometry - the structured, trustworthy source of truth
    # for per-diem unit counts (see DESIGN.md decision 4).
    @property
    def derived_nights(self) -> int:
        return (self.trip_end - self.trip_start).days

    @property
    def derived_days(self) -> int:
        return (self.trip_end - self.trip_start).days + 1

    @property
    def line_sum(self) -> Decimal:
        return money(sum((l.amount for l in self.lines), Decimal("0")))

    @property
    def days_to_submit(self) -> int:
        """POL-TIME-01 anchored on trip_start, the strictest defensible reading."""
        return (self.submitted_date - self.trip_start).days


# --- Appendix B, transcribed verbatim --------------------------------
CLAIMS: List[Claim] = [
    Claim(
        claim_id="CLM-001",
        title="Attend 2-day industry conference (business)",
        employee="A. Rivera",
        trip_start=date(2026, 6, 10),
        trip_end=date(2026, 6, 12),
        submitted_date=date(2026, 6, 20),
        stated_total=Decimal("1110.00"),
        lines=[
            LineItem(line_id="CLM-001-L1", category="airfare",
                     description="Round-trip economy airfare",
                     amount=Decimal("420.00"), receipt_attached=True),
            LineItem(line_id="CLM-001-L2", category="lodging",
                     description="Hotel, 2 nights @ $180",
                     amount=Decimal("360.00"), receipt_attached=True),
            LineItem(line_id="CLM-001-L3", category="meals",
                     description="Meals, 3 days @ ~$60/day",
                     amount=Decimal("180.00"), receipt_attached=True),
            LineItem(line_id="CLM-001-L4", category="conference_fees",
                     description="Conference registration",
                     amount=Decimal("150.00"), receipt_attached=True),
        ],
    ),
    Claim(
        claim_id="CLM-002",
        title="Weekend hotel stay",
        employee="B. Osei",
        trip_start=date(2026, 6, 14),
        trip_end=date(2026, 6, 15),
        submitted_date=date(2026, 6, 25),
        stated_total=Decimal("380.00"),
        lines=[
            LineItem(line_id="CLM-002-L1", category="spa",
                     description="Hotel spa package",
                     amount=Decimal("300.00"), receipt_attached=True),
            LineItem(line_id="CLM-002-L2", category="minibar",
                     description="In-room minibar",
                     amount=Decimal("80.00"), receipt_attached=True),
        ],
    ),
    Claim(
        claim_id="CLM-003",
        title="Client site visit (business)",
        employee="C. Nakamura",
        trip_start=date(2026, 6, 8),
        trip_end=date(2026, 6, 10),
        submitted_date=date(2026, 6, 22),
        stated_total=Decimal("940.00"),
        lines=[
            LineItem(line_id="CLM-003-L1", category="airfare",
                     description="Round-trip economy airfare",
                     amount=Decimal("300.00"), receipt_attached=True),
            LineItem(line_id="CLM-003-L2", category="lodging",
                     description="Hotel, 2 nights @ $250",
                     amount=Decimal("500.00"), receipt_attached=True),
            LineItem(line_id="CLM-003-L3", category="meals",
                     description="Meals, 2 days @ $70/day",
                     amount=Decimal("140.00"), receipt_attached=True),
        ],
    ),
    Claim(
        claim_id="CLM-004",
        title="International vendor negotiation (business)",
        employee="D. Fischer",
        trip_start=date(2026, 6, 16),
        trip_end=date(2026, 6, 18),
        submitted_date=date(2026, 6, 28),
        stated_total=Decimal("3000.00"),
        lines=[
            LineItem(line_id="CLM-004-L1", category="airfare",
                     description="Business-class international airfare",
                     amount=Decimal("2400.00"), receipt_attached=True),
            LineItem(line_id="CLM-004-L2", category="lodging",
                     description="Hotel, 3 nights",
                     amount=Decimal("600.00"), receipt_attached=False),
        ],
    ),
    Claim(
        claim_id="CLM-005",
        title="Client dinner / business development",
        employee="E. Haddad",
        trip_start=date(2026, 6, 11),
        trip_end=date(2026, 6, 11),
        submitted_date=date(2026, 6, 24),
        stated_total=Decimal("220.00"),
        lines=[
            LineItem(line_id="CLM-005-L1", category="meals",
                     description="Client dinner for 4 (business development)",
                     amount=Decimal("220.00"), receipt_attached=False),
        ],
    ),
]
CLAIMS_BY_ID = {c.claim_id: c for c in CLAIMS}


# =====================================================================
# SECTION 6 — DETERMINISTIC POLICY ENGINE
# =====================================================================
# Sole authority for eligibility, caps, unit derivation, totals, tiers,
# timeliness and conflicts. The LLM never computes any of this.
# =====================================================================
SEV_INFO, SEV_DEDUCT, SEV_BLOCK, SEV_CONFLICT = "info", "deduction", "blocker", "conflict"

# Informational findings that nonetheless DETERMINED an amount, so the rule
# behind them is genuinely load-bearing and earns a policy_refs citation.
# Everything else at info severity (TIMELY, NO_STATED_BUSINESS_PURPOSE, the
# non-blocking tier notes) was merely checked, and is not cited.
MATERIAL_INFO_CODES = {"WITHIN_PER_DIEM", "AIRFARE_ECONOMY", "ELIGIBLE_UNCAPPED"}

REASON_RULES = {
    "POLICY_EXCEPTION_AIRFARE_CLASS": ["POL-AIR-01"],
    "MISSING_REQUIRED_RECEIPT": ["POL-RCT-01", "POL-RCT-02"],
    "EXCEEDS_AGENT_AUTHORITY": ["POL-APR-03"],
    "LATE_SUBMISSION": ["POL-TIME-01"],
    "DATA_CONFLICT": [],
    "POLICY_SILENT": [],
    "UNKNOWN_CATEGORY": ["POL-CAT-01", "POL-CAT-02"],
    "LOW_CONFIDENCE": [],
    "AGENT_OUTPUT_INVALID": [],
}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    policy_refs: List[str] = field(default_factory=list)
    line_id: Optional[str] = None
    amount: Optional[Decimal] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "policy_refs": self.policy_refs,
        }
        if self.line_id:
            d["line_id"] = self.line_id
        if self.amount is not None:
            d["amount"] = str(money(self.amount))
        return d


@dataclass
class LineVerdict:
    line_id: str
    category: str
    description: str
    claimed: Decimal
    eligible: Optional[bool]
    allowed: Decimal
    excess: Decimal
    cap_applied: Optional[Decimal] = None
    units_used: Optional[Decimal] = None
    unit_kind: Optional[str] = None
    receipt_required: bool = False
    receipt_attached: bool = False
    policy_refs: List[str] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "category": self.category,
            "description": self.description,
            "claimed": str(money(self.claimed)),
            "eligible": self.eligible,
            "allowed": str(money(self.allowed)),
            "excess": str(money(self.excess)),
            "cap_applied": None if self.cap_applied is None else str(money(self.cap_applied)),
            "units_used": None if self.units_used is None else str(self.units_used),
            "unit_kind": self.unit_kind,
            "receipt_required": self.receipt_required,
            "receipt_attached": self.receipt_attached,
            "policy_refs": self.policy_refs,
            "findings": [f.as_dict() for f in self.findings],
        }


@dataclass
class RuleLedger:
    """Immutable-by-convention record of everything policy says about a claim."""
    claim_id: str
    lines: List[LineVerdict]
    findings: List[Finding]
    reimbursable_total: Decimal
    deducted_total: Decimal
    tier: str
    tier_rule: str
    within_agent_authority: bool
    days_to_submit: int
    timely: bool
    derived_nights: int
    derived_days: int
    candidate_decision: str
    reason_codes: List[str]
    missing_docs: List[str]
    policy_refs: List[str]

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity in (SEV_BLOCK, SEV_CONFLICT)]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "lines": [l.as_dict() for l in self.lines],
            "findings": [f.as_dict() for f in self.findings],
            "reimbursable_total": str(money(self.reimbursable_total)),
            "deducted_total": str(money(self.deducted_total)),
            "tier": self.tier,
            "tier_rule": self.tier_rule,
            "within_agent_authority": self.within_agent_authority,
            "days_to_submit": self.days_to_submit,
            "timely": self.timely,
            "derived_nights": self.derived_nights,
            "derived_days": self.derived_days,
            "candidate_decision": self.candidate_decision,
            "reason_codes": self.reason_codes,
            "missing_docs": self.missing_docs,
            "policy_refs": self.policy_refs,
        }


# Per-diem caps, read straight out of the registry so the rule text and the
# arithmetic can never drift apart.
def _cap_for(category: str) -> Optional[Tuple[Decimal, str, str]]:
    for rid in ("POL-PD-01", "POL-PD-02", "POL-PD-03"):
        r = REGISTRY.get(rid)
        if r and r.params.get("category") == category:
            return r.params["cap"], r.params["unit"], r.rule_id
    return None


class PolicyEngine:
    def __init__(self, registry: PolicyRegistry = REGISTRY):
        self.registry = registry

    # ---------- individual checks, each also exposed as a tool ----------
    def check_timeliness(self, claim: Claim) -> Tuple[bool, int, Finding]:
        rule = self.registry.get("POL-TIME-01")
        window = int(rule.params["window_days"])
        elapsed = claim.days_to_submit
        timely = elapsed <= window
        if timely:
            f = Finding(SEV_INFO, "TIMELY",
                        f"Submitted {elapsed} day(s) after trip start, within the "
                        f"{window}-day window.", ["POL-TIME-01"])
        else:
            f = Finding(SEV_BLOCK, "LATE_SUBMISSION",
                        f"Submitted {elapsed} day(s) after trip start, beyond the "
                        f"{window}-day window; POL-TIME-01 routes late claims to Manual Review.",
                        ["POL-TIME-01"])
        return timely, elapsed, f

    def check_receipts(self, claim: Claim) -> Tuple[List[Dict[str, Any]], List[str], List[Finding]]:
        rule = self.registry.get("POL-RCT-01")
        threshold = rule.params["threshold"]
        always = rule.params["always_required_categories"]
        rows, missing, findings = [], [], []
        for line in claim.lines:
            required = line.category in always or line.amount > threshold
            why = ("category always requires a receipt"
                   if line.category in always else
                   f"amount exceeds ${threshold}" if line.amount > threshold else "not required")
            ok = (not required) or line.receipt_attached
            rows.append({
                "line_id": line.line_id,
                "category": line.category,
                "amount": str(money(line.amount)),
                "receipt_required": required,
                "receipt_attached": line.receipt_attached,
                "compliant": ok,
                "reason": why,
            })
            if required and not line.receipt_attached:
                doc = (f"Itemized receipt for {line.category} line {line.line_id} "
                       f"(\"{line.description}\", ${money(line.amount)})")
                missing.append(doc)
                findings.append(Finding(
                    SEV_BLOCK, "MISSING_REQUIRED_RECEIPT",
                    f"{line.line_id} ({line.category}, ${money(line.amount)}) requires an "
                    f"itemized receipt ({why}) but none is attached. POL-RCT-02 routes the "
                    f"claim to Manual Review rather than rejecting the item.",
                    ["POL-RCT-01", "POL-RCT-02"], line.line_id, line.amount))
        return rows, missing, findings

    def check_threshold(self, reimbursable_total: Decimal) -> Tuple[str, str, bool, Finding]:
        t = money(reimbursable_total)
        r1, r2, r3 = (self.registry.get("POL-APR-01"),
                      self.registry.get("POL-APR-02"),
                      self.registry.get("POL-APR-03"))
        if t <= r1.params["upper"]:
            return ("auto_approve", "POL-APR-01", True,
                    Finding(SEV_INFO, "TIER_AUTO_APPROVE",
                            f"Reimbursable ${t} is within the ${r1.params['upper']} "
                            f"auto-approve tier.", ["POL-APR-01"]))
        if t <= r2.params["upper"]:
            return ("manager", "POL-APR-02", True,
                    Finding(SEV_INFO, "TIER_MANAGER",
                            f"Reimbursable ${t} falls in the manager tier "
                            f"(> ${r2.params['lower']}, <= ${r2.params['upper']}); "
                            f"approvable when fully compliant.", ["POL-APR-02"]))
        return ("director", "POL-APR-03", False,
                Finding(SEV_BLOCK, "EXCEEDS_AGENT_AUTHORITY",
                        f"Reimbursable ${t} exceeds ${r3.params['lower']}, beyond the agent's "
                        f"auto-approval authority; POL-APR-03 requires director approval via "
                        f"Manual Review.", ["POL-APR-03"]))

    def compute_limits(self, claim: Claim) -> Tuple[List[LineVerdict], List[Finding]]:
        verdicts: List[LineVerdict] = []
        claim_findings: List[Finding] = []

        for line in claim.lines:
            sig = line.signals
            v = LineVerdict(
                line_id=line.line_id, category=line.category, description=line.description,
                claimed=line.amount, eligible=None, allowed=money(0), excess=money(0),
                receipt_attached=line.receipt_attached,
            )

            # --- eligibility (POL-CAT-01 / POL-CAT-02) ---
            if line.category in INELIGIBLE_CATEGORIES:
                v.eligible = False
                v.allowed, v.excess = money(0), money(line.amount)
                v.policy_refs = ["POL-CAT-02"]
                v.findings.append(Finding(
                    SEV_DEDUCT, "INELIGIBLE_CATEGORY",
                    f"{line.category} is never reimbursable under POL-CAT-02; "
                    f"${money(line.amount)} deducted in full.",
                    ["POL-CAT-02"], line.line_id, line.amount))
                verdicts.append(v)
                continue

            if line.category not in ELIGIBLE_CATEGORIES:
                v.eligible = None
                v.allowed, v.excess = money(0), money(0)
                v.policy_refs = ["POL-CAT-01", "POL-CAT-02"]
                f = Finding(
                    SEV_BLOCK, "UNKNOWN_CATEGORY",
                    f"Category '{line.category}' appears in neither the POL-CAT-01 eligible "
                    f"list nor the POL-CAT-02 ineligible list; it cannot be adjudicated "
                    f"automatically.",
                    ["POL-CAT-01", "POL-CAT-02"], line.line_id, line.amount)
                v.findings.append(f)
                claim_findings.append(f)
                verdicts.append(v)
                continue

            v.eligible = True
            v.policy_refs = ["POL-CAT-01"]

            # --- airfare class (POL-AIR-01) ---
            if line.category == "airfare":
                v.policy_refs.append("POL-AIR-01")
                if sig.fare_class == "premium":
                    # Explicitly NOT auto-deducted: a pre-approval may exist.
                    v.allowed, v.excess = money(line.amount), money(0)
                    f = Finding(
                        SEV_BLOCK, "POLICY_EXCEPTION_AIRFARE_CLASS",
                        f"{line.line_id} is a non-economy fare (\"{line.description}\"). "
                        f"POL-AIR-01 makes this a policy exception routed to Manual Review, "
                        f"not an automatic deduction, because a pre-approval may exist.",
                        ["POL-AIR-01"], line.line_id, line.amount)
                    v.findings.append(f)
                    claim_findings.append(f)
                elif sig.fare_class == "economy":
                    v.allowed = money(line.amount)
                    v.findings.append(Finding(
                        SEV_INFO, "AIRFARE_ECONOMY",
                        f"Economy fare, reimbursable in full under POL-AIR-01.",
                        ["POL-AIR-01"], line.line_id, line.amount))
                else:
                    # Premium economy, or no class stated at all. POL-AIR-01
                    # reimburses economy ONLY, so neither can be approved on the
                    # strength of the data supplied. Not auto-deducted either -
                    # the fare may well be economy; a human has to confirm it.
                    v.allowed, v.excess = money(line.amount), money(0)
                    seen = ("stated as premium economy, which is neither economy nor "
                            "business/first class"
                            if sig.fare_class == "premium_economy"
                            else "does not state a fare class")
                    f = Finding(
                        SEV_BLOCK, "AIRFARE_CLASS_UNVERIFIED",
                        f"{line.line_id} (\"{line.description}\") {seen}. POL-AIR-01 "
                        f"reimburses economy class only, so eligibility cannot be "
                        f"established from the claim as submitted; a reviewer must "
                        f"confirm the class before any amount is authorised.",
                        ["POL-AIR-01"], line.line_id, line.amount)
                    v.findings.append(f)
                    claim_findings.append(f)
                verdicts.append(v)
                continue

            # --- per-diem capped categories (POL-PD-01/02/03) ---
            capinfo = _cap_for(line.category)
            if capinfo is None:
                # Eligible, uncapped (conference_fees).
                v.allowed = money(line.amount)
                v.findings.append(Finding(
                    SEV_INFO, "ELIGIBLE_UNCAPPED",
                    f"{line.category} is eligible under POL-CAT-01 and carries no per-diem cap.",
                    ["POL-CAT-01"], line.line_id, line.amount))
                verdicts.append(v)
                continue

            cap_rate, unit, cap_rule = capinfo
            v.policy_refs.append(cap_rule)
            derived = Decimal(claim.derived_nights if unit == "night" else claim.derived_days)

            # Trip dates are authoritative; the description's own quantity is a
            # cross-check (DESIGN.md decision 4).
            if sig.stated_units is not None and sig.stated_unit_kind == unit:
                units = min(sig.stated_units, derived)
                if sig.stated_units > derived:
                    f = Finding(
                        SEV_CONFLICT, "DATA_CONFLICT",
                        f"{line.line_id} states {sig.stated_units} {unit}(s) but the trip dates "
                        f"{claim.trip_start} to {claim.trip_end} imply {derived}. Trip dates are "
                        f"authoritative, so the cap is computed on {derived} {unit}(s); the "
                        f"discrepancy needs a human to reconcile.",
                        [cap_rule], line.line_id, line.amount)
                    v.findings.append(f)
                    claim_findings.append(f)
            else:
                units = derived

            if units <= 0:
                if unit == "day":
                    # A same-day trip is still one day of subsistence.
                    units = Decimal(1)
                else:
                    # Nights are different: a trip that starts and ends on the
                    # same date has no nights, so there is no lodging allowance
                    # to compute. Granting a night's cap here would authorise
                    # money for a stay the dates say never happened.
                    units = Decimal(0)
                    f = Finding(
                        SEV_CONFLICT, "DATA_CONFLICT",
                        f"{line.line_id} claims {line.category} of ${money(line.amount)} but "
                        f"the trip dates {claim.trip_start} to {claim.trip_end} span zero "
                        f"{unit}(s), so no {unit}ly allowance exists under {cap_rule}. The "
                        f"dates and the claim contradict each other and a human must "
                        f"reconcile them.",
                        [cap_rule], line.line_id, line.amount)
                    v.findings.append(f)
                    claim_findings.append(f)

            cap_total = money(cap_rate * units)
            v.units_used, v.unit_kind, v.cap_applied = units, unit, cap_total
            v.allowed = money(min(line.amount, cap_total))
            v.excess = money(line.amount - v.allowed)

            if v.excess > 0:
                v.findings.append(Finding(
                    SEV_DEDUCT, "OVER_PER_DIEM",
                    f"{line.line_id} claims ${money(line.amount)} against a "
                    f"${cap_total} cap ({units} {unit}(s) x ${cap_rate}); "
                    f"${v.excess} deducted under {cap_rule}.",
                    [cap_rule], line.line_id, v.excess))
            else:
                v.findings.append(Finding(
                    SEV_INFO, "WITHIN_PER_DIEM",
                    f"{line.line_id} claims ${money(line.amount)} against a ${cap_total} cap "
                    f"({units} {unit}(s) x ${cap_rate}); within limit.",
                    [cap_rule], line.line_id, line.amount))

            # --- policy silence on multi-person meals ---
            if line.category == "meals" and sig.headcount and sig.headcount > 1:
                f = Finding(
                    SEV_BLOCK, "POLICY_SILENT",
                    f"{line.line_id} covers {sig.headcount} people "
                    f"(\"{line.description}\"). POL-PD-01 caps meals at ${cap_rate} per day and "
                    f"is silent on headcount and on client entertainment, so no per-person "
                    f"scaling is authorised. The ${cap_total} cap is applied for the claimant "
                    f"alone and the ambiguity is referred to a reviewer.",
                    ["POL-PD-01", "POL-CAT-01"], line.line_id, line.amount)
                v.findings.append(f)
                claim_findings.append(f)

            verdicts.append(v)

        return verdicts, claim_findings

    def detect_conflicts(self, claim: Claim) -> List[Finding]:
        """Claim-level integrity checks beyond the per-line ones."""
        out: List[Finding] = []
        if claim.line_sum != money(claim.stated_total):
            out.append(Finding(
                SEV_CONFLICT, "DATA_CONFLICT",
                f"Line items sum to ${claim.line_sum} but the claim states a total of "
                f"${money(claim.stated_total)}; the arithmetic does not reconcile.",
                []))
        if not re.search(r"business|client|conference|vendor|work", claim.title, re.I):
            out.append(Finding(
                SEV_INFO, "NO_STATED_BUSINESS_PURPOSE",
                f"The claim title \"{claim.title}\" states no business purpose. POL-CAT-01 "
                f"reimburses only expenses incurred for a documented business purpose.",
                ["POL-CAT-01"]))
        return out

    # ---------------------- full adjudication ----------------------
    def adjudicate(self, claim: Claim) -> RuleLedger:
        line_verdicts, limit_findings = self.compute_limits(claim)
        receipt_rows, missing_docs, receipt_findings = self.check_receipts(claim)
        timely, elapsed, time_finding = self.check_timeliness(claim)
        conflict_findings = self.detect_conflicts(claim)

        for v in line_verdicts:
            for row in receipt_rows:
                if row["line_id"] == v.line_id:
                    v.receipt_required = row["receipt_required"]

        reimbursable = money(sum((v.allowed for v in line_verdicts), Decimal("0")))
        deducted = money(sum((v.excess for v in line_verdicts), Decimal("0")))
        tier, tier_rule, authority, tier_finding = self.check_threshold(reimbursable)

        findings: List[Finding] = []
        for v in line_verdicts:
            findings.extend(v.findings)
        findings.extend(limit_findings)
        findings.extend(receipt_findings)
        findings.append(time_finding)
        findings.extend(conflict_findings)
        findings.append(tier_finding)

        # de-duplicate findings that were recorded both per-line and per-claim
        seen, deduped = set(), []
        for f in findings:
            key = (f.severity, f.code, f.message, f.line_id)
            if key not in seen:
                seen.add(key)
                deduped.append(f)
        findings = deduped

        # ---------------- precedence ladder (DESIGN.md §6) ----------------
        has_lines = len(line_verdicts) > 0
        all_ineligible = has_lines and all(v.eligible is False for v in line_verdicts)
        blockers = [f for f in findings if f.severity in (SEV_BLOCK, SEV_CONFLICT)]

        if all_ineligible:
            decision = "REJECT"
        elif blockers:
            decision = "MANUAL_REVIEW"
        elif deducted > 0:
            decision = "PARTIAL_APPROVE"
        else:
            decision = "APPROVE"

        reason_codes = sorted({f.code for f in blockers}) if decision == "MANUAL_REVIEW" else []

        # ---- policy_refs: rules that DROVE this decision, not every rule read ----
        # Citing each of the twelve rules merely because it was checked turns
        # policy_refs into noise - POL-TIME-01 on a claim that was comfortably
        # on time tells a reviewer nothing. A rule earns a citation when it
        # changed an amount, blocked the claim, or authorised the approval.
        refs: List[str] = []

        def cite(rule_id: str) -> None:
            if rule_id not in refs and self.registry.exists(rule_id):
                refs.append(rule_id)

        for f in findings:
            if f.severity != SEV_INFO or f.code in MATERIAL_INFO_CODES:
                for r in f.policy_refs:
                    cite(r)

        # Eligibility underpins every line that was paid or struck out.
        if any(v.eligible is True for v in line_verdicts):
            cite("POL-CAT-01")
        if any(v.eligible is False for v in line_verdicts):
            cite("POL-CAT-02")

        # The approval tier is material when it authorises the payment, or when
        # it is itself the blocker (POL-APR-03). On a REJECT or an escalation
        # for other reasons, the tier never came into play.
        if decision in ("APPROVE", "PARTIAL_APPROVE") or not authority:
            cite(tier_rule)

        refs = sorted(refs)

        return RuleLedger(
            claim_id=claim.claim_id,
            lines=line_verdicts,
            findings=findings,
            reimbursable_total=reimbursable,
            deducted_total=deducted,
            tier=tier,
            tier_rule=tier_rule,
            within_agent_authority=authority,
            days_to_submit=elapsed,
            timely=timely,
            derived_nights=claim.derived_nights,
            derived_days=claim.derived_days,
            candidate_decision=decision,
            reason_codes=reason_codes,
            missing_docs=missing_docs,
            policy_refs=refs,
        )


ENGINE = PolicyEngine()
