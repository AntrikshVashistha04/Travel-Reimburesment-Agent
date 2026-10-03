"""Assemble AntrikshVashistha.ipynb from the verified modules.

Every code cell below is lifted verbatim from a module that was built and
tested standalone first; the only transformation is stripping the
cross-module imports that existed purely so the modules could be tested
separately. In the notebook everything shares one namespace.
"""
import json
import re
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE.parent / "AntrikshVashistha.ipynb"      # repo root, beside this package

LOCAL_MODULES = ("core", "core2", "core3", "dash", "evals")


def strip_local_imports(text: str) -> str:
    """Remove `from core import ...` style lines, including multi-line ones,
    and the __future__ import (every annotation is runtime-valid on 3.9)."""
    out, lines, i = [], text.splitlines(keepends=True), 0
    # Leading whitespace is allowed: these also appear as function-local imports
    # inside the modules, and in the notebook every name is already global.
    pat = re.compile(r"^\s*from (%s) import" % "|".join(LOCAL_MODULES))
    while i < len(lines):
        line = lines[i]
        if line.startswith("from __future__ import"):
            i += 1
            continue
        if pat.match(line):
            depth = line.count("(") - line.count(")")
            i += 1
            while depth > 0 and i < len(lines):
                depth += lines[i].count("(") - lines[i].count(")")
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out).strip("\n")


def sections(module: str):
    """Split a module on its `# ==== / # SECTION ...` banners."""
    text = strip_local_imports((HERE / f"{module}.py").read_text())
    marks = [m.start() for m in
             re.finditer(r"^# ={20,}\n# SECTION ", text, re.M)]
    marks.append(len(text))
    return [text[marks[i]:marks[i + 1]].strip("\n") for i in range(len(marks) - 1)]


C = sections("core")        # 3: policy | 4: claims | 6: engine
C2 = sections("core2")      # 5: contract | 7: tools | 8: agent | 9: gate
C3 = sections("core3")      # modes/model | orchestrator
D = sections("dash")        # dashboard | review console
E = sections("evals")       # eval harness | ablation

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))


# =====================================================================
md(r"""
# Travel Reimbursement Approval Agent

**Antriksh Vashistha** · A GenAI/Agentic AI prototype that adjudicates employee travel
reimbursement claims against the Appendix A policy and returns a structured recommendation
for each of the five Appendix B claims.

---

## The one idea this notebook is built on

> **A deterministic policy engine owns every dollar and every hard rule.
> The LLM owns retrieval, tool selection, synthesis and explanation.
> They meet at a reconciliation gate that the LLM cannot win.**

The agent does real agentic work — it is handed the claim **but not the policy**, so it must
retrieve the rules itself, choose which checks to run, combine their results and resolve
conflicts. What it cannot do is produce a wrong number: the ledger is computed before the
agent runs and the gate overwrites the agent's arithmetic afterwards. That is the difference
between *hoping* the model does not miscalculate and *knowing* it cannot.

Escalation is **monotonic**: any layer may route a claim to `MANUAL_REVIEW`, no layer may
relax one. That single property is what makes "prefer Manual Review over forcing a decision"
structurally true rather than dependent on how the prompt was worded.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Python 3.9+ (3.11+ recommended)
pip install -r requirements.txt                          # or let section 1 install them
jupyter lab AntrikshVashistha.ipynb                      # then Run All
```

The notebook **runs top to bottom with no manual steps and no API key.** Section 1 installs
anything missing.

### Environment variables

| Variable | Required | Default | Effect |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | no | unset | When set, the agent calls Claude for real (`LIVE` mode). |
| `REIMB_MODEL` | no | `claude-sonnet-5` | Model id used in `LIVE` mode. |
| `REIMB_MODE` | no | auto | Force `LIVE`, `REPLAY` or `SIMULATED`. |

### The three execution modes

| Mode | When | What drives tool selection |
|---|---|---|
| `LIVE` | `ANTHROPIC_API_KEY` is set | Claude, through LangChain's `AgentExecutor`. |
| `REPLAY` | no key, `transcripts.json` present | AI turns recorded from a previous live run. |
| `SIMULATED` | no key, no transcript | A deterministic stand-in. **Not a language model and not a recording of one** — it drives the same executor and the same tools, but its tool choices follow a fixed plan and its prose is template-generated. |

**The five results are identical in all three modes**, because the decision and every amount
come from the deterministic engine, never from the model. What changes between modes is only
*who chose which tools to call* and *who wrote the explanation prose*. The mode is printed in
the banner in section 2 and carried on every result card, so you always know what you are
looking at.

To record a real transcript once and make `REPLAY` available to reviewers without a key:

```python
from pathlib import Path
record_transcripts(Path("transcripts.json"))   # needs ANTHROPIC_API_KEY
```

### What the notebook produces

* a JSON array of five result objects (the **final code cell**), each with exactly the nine
  required fields
* a dashboard under **## Dashboard**, exported as `UI SS_1.png`
* a per-claim review console, exported as `UI SS_2.png`
* `outputs/decisions.json` and `outputs/audit_trail.json`

### Map of the notebook

| § | What |
|---|---|
| 1–2 | Dependencies, configuration, execution mode |
| 3–4 | Appendix A as a typed rule registry · Appendix B as validated claims |
| 5 | The nine-field output contract |
| 6 | The deterministic policy engine |
| 7–8 | Six tools · the LangChain agent loop and the model layer |
| 9 | Reconciliation gate and confidence scorer |
| 10–11 | Run all five · three worked walkthroughs |
| 12 | **Dashboard** and review console |
| 13–14 | Evaluation harness · ablation |
| 15–17 | Design notes · assumptions and limitations · demo evidence |
| final | The JSON array of five results |
""")

# ---------------------------------------------------------------- 1
md("## 1. Dependencies\n\nIdempotent: installs only what is missing, so re-running is cheap.")
code(r'''
import importlib.util, subprocess, sys

REQUIRED = {
    "pydantic":            "pydantic>=2.5,<3",
    "pandas":              "pandas>=2.0",
    "matplotlib":          "matplotlib>=3.7",
    "langchain":           "langchain>=0.3,<0.4",
    "langchain_core":      "langchain-core>=0.3,<0.4",
    "langchain_anthropic": "langchain-anthropic>=0.3,<0.4",
    "ipywidgets":          "ipywidgets>=8.0",
}
missing = [spec for mod, spec in REQUIRED.items()
           if importlib.util.find_spec(mod) is None]
if missing:
    print("installing:", ", ".join(missing))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("all dependencies present")

assert sys.version_info >= (3, 9), "Python 3.9+ required (3.11+ recommended)"
print("python", ".".join(map(str, sys.version_info[:3])))
''')

# ---------------------------------------------------------------- 2
md(r"""
## 2. Configuration and execution mode

Note the **frozen clock**. Today's date is irrelevant to this notebook: POL-TIME-01 is
evaluated from each claim's own `submitted_date` against its `trip_start`. A stray
`datetime.now()` would make all five June-2026 claims read as late and silently corrupt every
result, so section 13 asserts the elapsed-day counts explicitly.
""")
code(r'''
import json, os, warnings
from decimal import Decimal
warnings.filterwarnings("ignore")

MODEL_NAME = os.environ.get("REIMB_MODEL", "claude-sonnet-5")
HAS_KEY    = bool(os.environ.get("ANTHROPIC_API_KEY"))
FORCED     = os.environ.get("REIMB_MODE", "").strip().upper() or "(auto)"

print(f"ANTHROPIC_API_KEY : {'set' if HAS_KEY else 'not set'}")
print(f"REIMB_MODEL       : {MODEL_NAME}")
print(f"REIMB_MODE        : {FORCED}")
print("clock             : frozen - timeliness uses each claim's own dates, never now()")
''')

# ---------------------------------------------------------------- 3
md(r"""
## 3. Appendix A as code — the policy registry

All twelve `POL-*` rules, transcribed **verbatim**. Each carries the policy's own wording in
`text` plus a machine-readable `params` projection the engine computes with. Nothing is
invented: every number in `params` appears in the `text` beside it.

Appendix A's closing *Decision Guidance* block carries no `POL-*` id of its own, so it is held
in a separate constant and never emitted as a `policy_ref` — doing otherwise would mean
inventing an identifier.

Retrieval is exact keyword and category matching over twelve typed records. **No vector
store**: with a one-page policy, embeddings would buy fuzzy-match failure modes,
non-determinism and a dependency in exchange for nothing.
""")
code(C[0])
code(r'''
import pandas as pd
pd.set_option("display.max_colwidth", 78)

display(pd.DataFrame([{
    "rule_id": r.rule_id, "type": r.rule_type, "title": r.title,
    "params": ", ".join(f"{k}={v}" for k, v in r.params.items())[:70] or "-",
} for r in POLICY_RULES]))

print(f"{len(REGISTRY)} rules loaded: {', '.join(REGISTRY.ids)}\n")
print("Retrieval demo - REGISTRY.search('hotel nightly cap'):")
for r in REGISTRY.search("hotel nightly cap")[:3]:
    print(f"  {r.rule_id}  {r.title}")
''')

# ---------------------------------------------------------------- 4
md(r"""
## 4. Appendix B as data — the five claims

The five claims, transcribed verbatim and validated by Pydantic at the boundary.

One detail worth pausing on. Appendix B states quantities *inside the description string*
("Hotel, 2 nights @ $180", "Client dinner for 4"). Rather than hand-enter those numbers, the
engine **parses them out of the supplied text** — so the extraction is part of the
demonstrable pipeline, and the CLM-004 conflict surfaces from the data rather than from an
assumption typed in by me.
""")
code(C[1])
code(r'''
rows = []
for c in CLAIMS:
    for l in c.lines:
        s = l.signals
        rows.append({
            "claim": c.claim_id, "line": l.line_id, "category": l.category,
            "amount": float(l.amount), "receipt": l.receipt_attached,
            "parsed_units": f"{s.stated_units} {s.stated_unit_kind}" if s.stated_units else "-",
            "parsed_rate": f"${s.stated_rate}" if s.stated_rate else "-",
            "headcount": s.headcount or "-", "fare_class": s.fare_class or "-",
        })
display(pd.DataFrame(rows))

print("Trip geometry derived from the claim's own dates:")
for c in CLAIMS:
    print(f"  {c.claim_id}  {c.trip_start} -> {c.trip_end}   "
          f"{c.derived_nights} night(s), {c.derived_days} day(s)   "
          f"submitted +{c.days_to_submit}d   lines sum ${c.line_sum} "
          f"vs stated ${c.stated_total}")
''')

# ---------------------------------------------------------------- 5
md(r"""
## 5. The output contract

Exactly nine fields, enforced with `extra="forbid"`. The model-level validator encodes the
invariants that make a result self-consistent: `APPROVE` cannot carry a deduction, `REJECT`
cannot carry an approval, `PARTIAL_APPROVE` needs both, a non-empty `missing_docs` forces
`MANUAL_REVIEW`, and every `policy_refs` entry must resolve to a real Appendix A rule.

`policy_refs` carries the rules that **drove** the decision, not every rule that was read. A
rule earns its place by changing an amount, blocking the claim, or authorising the approval.
Citing POL-TIME-01 on a claim submitted comfortably inside the window would tell a reviewer
nothing — so a rule that was merely checked and passed is not cited. On a *late* claim
POL-TIME-01 becomes a blocker and is cited.

The full audit trail lives in a **separate parallel object**. The brief says *exactly* these
nine fields, and that is honoured literally.
""")
code(C2[0])

# ---------------------------------------------------------------- 6
md(r"""
## 6. The deterministic policy engine

Sole authority for eligibility, caps, unit derivation, totals, tiers, timeliness and
conflicts. Pure Python, `Decimal` throughout, no LLM anywhere near it.

**Per-diem units come from the trip dates, not from the description.** The stated quantity is
kept as a cross-check: when the two disagree the engine computes on the dates and raises
`DATA_CONFLICT` rather than silently picking one. CLM-004 says "Hotel, 3 nights" but its dates
span two — that is a genuine contradiction in the supplied data and a human should resolve it.

### Precedence ladder

```
1. classify each line            POL-CAT-01 / POL-CAT-02 / unknown
2. ALL lines ineligible       -> REJECT           deduct in full
3. collect blockers:             non-economy airfare (AIR-01)
                                 missing required receipt (RCT-01/02)
                                 late submission (TIME-01)
                                 reimbursable > $2,000 (APR-03)
                                 any conflict - unknown category
4. any blocker                -> MANUAL_REVIEW
5. deductions > 0             -> PARTIAL_APPROVE
6. otherwise                  -> APPROVE           tier APR-01 or APR-02
```

Step 2 precedes step 4 deliberately: the Decision Guidance defines rejection as *"the claimed
items are ineligible with nothing reimbursable"*, and there is nothing for a human to review
in a claim with no reimbursable content.
""")
code(C[2])
code(r'''
# The engine standing on its own, before any LLM is involved.
led = ENGINE.adjudicate(CLAIMS_BY_ID["CLM-003"])
print(f"CLM-003 -> {led.candidate_decision}   reimbursable ${led.reimbursable_total}   "
      f"deducted ${led.deducted_total}   tier {led.tier} ({led.tier_rule})\n")
display(pd.DataFrame([{
    "line": v.line_id, "category": v.category, "claimed": float(v.claimed),
    "cap": float(v.cap_applied) if v.cap_applied else None,
    "units": f"{v.units_used} {v.unit_kind}" if v.units_used else "-",
    "allowed": float(v.allowed), "excess": float(v.excess),
} for v in led.lines]))
for f in led.findings:
    print(f"  [{f.severity:9}] {f.code:28} {f.message[:96]}")
''')

# ---------------------------------------------------------------- 7
md(r"""
## 7. The tool layer

Six tools, against a stated minimum of two. Each wraps a real engine function, each cites the
rules it applied, each is independently testable. Tools return structured `findings` with a
`severity` rather than raising, so the agent can reason about a problem instead of crashing on
it.

`submit_decision` is the seventh: it carries `return_direct=True`, so the agent's final answer
arrives schema-shaped rather than as JSON buried in prose.
""")
code(C2[1])

# ---------------------------------------------------------------- 8
md(r"""
## 8. The agent loop and the model layer

LangChain `AgentExecutor` with `create_tool_calling_agent`.

The system prompt is where the agentic contract is set: the agent is told it has **not** been
given the policy and must retrieve it, and that it must never compute money itself. Withholding
the policy from the prompt is what makes grounding demonstrable rather than merely asserted —
if the agent never called `policy_lookup`, it would have nothing to cite.

Claim descriptions are explicitly marked as untrusted employee-supplied data, never as
instructions to the agent.

Failure handling distinguishes two different events, by a **typed** `failure_kind` rather than
by matching substrings of an error message:

| What happened | Treatment |
|---|---|
| Draft rejected by the schema, or never submitted | One **repair turn**: the validation error is handed back to the agent and it tries again. Still bad → `MANUAL_REVIEW` at confidence `0.0`. |
| The model could not be reached | No retry (pointless). Falls back to the engine-only decision, flagged `LLM_UNAVAILABLE`, with confidence reduced. |

An unusable answer and an unreachable model deserve opposite responses — escalate the first,
fall back on the second — so conflating them would be a real defect. `AgentDraft` is
deliberately strict on `decision` and `explanation` for the same reason: if every payload
validated, the repair turn could never fire.

The pipeline never crashes and never silently approves.
""")
code(C2[2])
md("""### The model layer — LIVE, REPLAY and SIMULATED

All three satisfy the same `BaseChatModel` interface, so a single code path serves every mode
and there is no separate "offline pipeline" to drift out of sync.""")
code(C3[0])

# ---------------------------------------------------------------- 9
md(r"""
## 9. Reconciliation gate and confidence scorer

Where the architecture is actually enforced. The gate takes the agent's draft and the engine's
ledger and produces the final answer:

* **amounts** always come from the ledger;
* **the decision** comes from the ledger, except that the agent may *escalate* to
  `MANUAL_REVIEW` — escalation is honoured, relaxation is refused and logged;
* **policy refs** are intersected with the real registry, and anything invented is dropped and
  recorded;
* **the explanation is validated, not passed through** — see below;
* **confidence** is a deterministic rubric, never the model's self-report.

### Why the explanation is validated too

The explanation is the field a human actually reads, and it was the one place a wrong model
answer could still reach them: the gate would correct the decision and the amounts while the
prose — written to justify the verdict that had just been *rejected* — shipped unchanged. A
claim could ship as `REJECT $0.00` carrying the sentence *"spa and minibar are wellness
benefits and fully reimbursable"*.

`validate_explanation()` now rejects prose that (a) argues for a decision other than the one
shipped, (b) asserts a dollar figure that appears nowhere in the ledger, (c) cites a `POL-*`
id that does not exist, or (d) runs past the five-sentence limit the system prompt sets.
Anything rejected falls back to the deterministic explanation and costs 0.10 of confidence.
Failing this check is safe by construction — the fallback is always available — so the check
is deliberately strict.

### What `confidence` means here

**Confidence that the returned `decision` is correct — including the decision to escalate.**
So a clear-cut `MANUAL_REVIEW` scores *high*: we are very sure a human is needed.

The alternative reading — confidence that *automated adjudication* is safe, under which every
escalation scores low — is defensible, and I considered it. I rejected it because uncertainty
about the money is already expressed by routing to `MANUAL_REVIEW`; encoding it a second time
in the confidence field makes "high-confidence escalation" inexpressible, and that is a
genuinely useful thing to be able to say. Hence unambiguous blockers (a missing receipt, a
$2,000 breach) *raise* confidence, while interpretive ones (`POLICY_SILENT`, `DATA_CONFLICT`)
lower it. CLM-004 lands at 0.92; CLM-005 at 0.72, because policy silence is murkier than an
absent receipt.

The model's self-reported confidence is captured and may only *lower* the score, never raise
it — a cheap way to use a weak signal without trusting it.
""")
code(C2[3])

# ---------------------------------------------------------------- 10
md("## 10. Orchestrator — run all five claims")
code(C3[1])
code(r'''
MODE = detect_mode()
print("=" * 78)
print(f"EXECUTION MODE: {MODE}")
print("=" * 78)
print(MODE_BANNERS[MODE])
print("=" * 78, "\n")

RUNS = adjudicate_all(mode=MODE)

display(pd.DataFrame([{
    "claim_id": r.result["claim_id"],
    "decision": r.result["decision"],
    "approved": r.result["approved_amount"],
    "deducted": r.result["deducted_amount"],
    "confidence": r.result["confidence"],
    "missing_docs": len(r.result["missing_docs"]),
    "policy_refs": len(r.result["policy_refs"]),
    "tool_calls": len(r.result["tools_used"]),
} for r in RUNS]))
''')

# ---------------------------------------------------------------- 11
md(r"""
## 11. Sample outputs — three worked walkthroughs

Full evidence for three claims that exercise different paths: a clean approval, a per-diem
deduction, and an escalation with four independent triggers. Each shows the rules the agent
retrieved, every tool call it made, the ledger the engine produced, and what the gate changed.
""")
code(r'''
def walkthrough(claim_id):
    r = next(x for x in RUNS if x.claim.claim_id == claim_id)
    c, led, d = r.claim, r.ledger, r.result
    print("=" * 94)
    print(f"{c.claim_id}  {c.title}")
    print(f"{c.employee}  ·  trip {c.trip_start} to {c.trip_end}  ·  "
          f"submitted {c.submitted_date}  ·  claimed ${c.stated_total}")
    print("=" * 94)

    print(f"\n1. CONTEXT RETRIEVED  ({len(r.audit.retrieved_rules)} rules)")
    print(f"   {', '.join(r.audit.retrieved_rules)}")

    print(f"\n2. TOOL TRACE  ({len(r.audit.calls)} calls)")
    for i, call in enumerate(r.audit.calls, 1):
        arg = json.dumps(call.args)
        print(f"   {i}. {call.name:28} {arg[:58]:60} {call.ms:>4}ms")

    print("\n3. ENGINE LEDGER")
    for v in led.lines:
        cap = f"cap ${v.cap_applied} ({v.units_used} {v.unit_kind})" if v.cap_applied else ""
        print(f"   {v.line_id:14} {v.category:16} claimed ${str(v.claimed):>8}  "
              f"allowed ${str(v.allowed):>8}  excess ${str(v.excess):>7}  {cap}")
    print(f"   {'':14} {'TOTAL':16} reimbursable ${led.reimbursable_total}  "
          f"deducted ${led.deducted_total}  tier {led.tier} ({led.tier_rule})")

    print("\n4. FINDINGS")
    for f in led.findings:
        if f.severity != "info":
            print(f"   [{f.severity:9}] {f.code:30} {', '.join(f.policy_refs)}")

    print("\n5. RECONCILIATION GATE")
    print(f"   engine said {r.gate.engine_decision}; agent said {r.gate.llm_decision}; "
          f"disagreement: {r.gate.disagreed}")
    for n in (r.gate.notes or ["   no override needed"]):
        print(f"   - {n}")
    print(f"   confidence trace: {' | '.join(r.gate.confidence_notes)}")

    print("\n6. RESULT")
    print(json.dumps(d, indent=2))
    if r.gate.packet:
        print("\n7. MANUAL REVIEW PACKET")
        print(json.dumps(r.gate.packet.as_dict(), indent=2))
    print()

walkthrough("CLM-001")
''')
code('walkthrough("CLM-003")')
code('walkthrough("CLM-004")')

# ---------------------------------------------------------------- 12
md(r"""
## Dashboard

Derived entirely from the actual claim outcomes above — nothing here is hard-coded.

A note on the colour language, since it is doing real work. The four decisions are *states*,
so they use one fixed, validated palette throughout: **green = approved**, **blue = awaiting a
human**, **amber = partially approved**, **red = deducted or rejected**. The palette was run
through a contrast and colour-vision-deficiency validator; green is never placed adjacent to
red in a stack (they measure ΔE 4.1 under deuteranopia — the classic red/green confusion), so
blue always separates them. Every mark is directly labelled and a full table is printed
alongside, so colour never carries meaning on its own.

With five claims, a decision "breakdown" is four counts — that is a stat-tile row, not a chart.
""")
code(D[0])
code(r'''
import matplotlib.pyplot as plt
fig, DF = build_dashboard(RUNS, save_to="UI SS_1.png")
plt.show()
display(DF[["claim_id", "employee", "decision", "claimed", "approved",
            "deducted", "awaiting_review", "confidence"]])
''')
md(r"""
### Review console

The interface a reviewer actually drives: pick a claim, read the recommendation, the
line-by-line ledger, the tool trace and what the gate changed. It uses `ipywidgets` when
available and falls back to static cards otherwise — on GitHub, under `nbconvert`, or in a
plain Jupyter install — so the notebook never depends on the widget extension being present.
""")
code(D[1])
code(r'''
console = review_console(RUNS)
''')
md("""GitHub's notebook renderer does not execute widget JavaScript, so the panel above is blank
there. The same five cards are rendered statically below, which is what a reviewer reading the
committed notebook on GitHub actually sees.""")
code(r'''
from IPython.display import HTML, display
for r in RUNS:
    display(HTML(result_html(r)))
''')
code(r'''
# Export the single-claim review card as the second UI screenshot.
card = build_claim_card(next(r for r in RUNS if r.claim.claim_id == "CLM-004"),
                        save_to="UI SS_2.png")
plt.show()
''')

# ---------------------------------------------------------------- 13
md(r"""
## 13. Evaluation harness

Ten checks. The golden outcomes were derived by hand from Appendix A *before* any code was
written, which is what makes them a test rather than a restatement of whatever the code
happens to do.

The robustness fixture is a deliberately corrupt claim — unknown category, unreconciled total,
121 days late. It is **a test fixture, not a sixth claim**: it never enters the five results or
the dashboard.
""")
code(E[0])
code(r'''
EVAL = run_evaluation(RUNS, MODE)
display(EVAL.frame())
print("\nGolden outcomes, hand-derived from Appendix A:")
display(EVAL.golden_df)
print("\nALL CHECKS PASSED" if EVAL.all_passed else "\nSOME CHECKS FAILED")
''')

# ---------------------------------------------------------------- 14
md(r"""
## 14. Ablation — what the reconciliation gate is actually for

It is easy to *claim* the gate matters. This measures it.

A deliberately wrong draft is pushed through the gate: it approves CLM-004 in full, reports no
missing documents, cites two invented policy ids, and is 99% sure of itself. This is exactly
the failure mode of an ungrounded LLM given a plausible-sounding claim. The comparison shows
what shipped instead.

This ablation needs no API key. The second one — the true LLM-only comparison, where a real
model adjudicates with no tools, no policy and no engine — only means anything against a live
model, so it is skipped with an explicit notice off `LIVE` rather than faked.
""")
code(E[1])
code(r'''
AB = run_gate_ablation()
display(AB["comparison"])
print(f"\nmoney the gate prevented being authorised: ${AB['money_at_risk']:,.2f}")
print(f"invented policy ids dropped: {AB['fake_refs_dropped']}")
print("\ngate actions:")
for n in AB["gate_notes"]:
    print("  -", n)
''')
code(r'''
LLM_AB = run_llm_only_ablation(MODE)
if LLM_AB.get("skipped"):
    print("LLM-only ablation skipped:", LLM_AB["reason"])
else:
    display(LLM_AB["comparison"])
    print("ungrounded LLM accuracy against the golden table:", LLM_AB["accuracy"])
''')

# ---------------------------------------------------------------- 15
md(r"""
## 15. Design Notes & Reasoning

### The central trade-off

The brief asks for an *agentic* system whose business decisions are *correct*. Those pull in
opposite directions: the thing that makes an agent useful — free choice over what to do next —
is the thing that makes it unreliable with money.

I resolved it by splitting the problem rather than compromising on either half. The LLM gets
full freedom over the parts where freedom helps: which rules to retrieve, which checks to run
and in what order, how to reconcile their results, how to explain the outcome. It gets zero
authority over the parts where freedom hurts: the arithmetic and the final decision. The
ledger is computed *before* the agent runs; the gate overwrites the agent's numbers *after*.

The alternative — let the LLM decide and check it afterwards — fails open. If a check is
missing, a wrong number ships. This design fails closed: to ship a wrong number you would have
to break the engine, and the engine is 200 lines of straight-line Python with a golden test.

### Why the agent is not given the policy

The agent is handed the claim JSON and nothing else. It must call `policy_lookup` before it
can cite anything. This costs a round-trip and buys something important: grounding becomes
*demonstrable*. Section 11 shows the rules each claim actually retrieved. Had I pasted the
policy into the system prompt, "the agent used the policy" would be an assertion with no
evidence behind it.

### Why no vector database

Twelve rules, roughly 800 words. A typed registry with keyword and category lookup is exact,
auditable, instant and has no failure mode. Embeddings would add a dependency, a similarity
threshold to tune, and the possibility of silently retrieving the wrong rule. The honest answer
to *"how would this scale?"* is that at a 200-page policy this inverts completely and retrieval
earns its place — but building it here would be cargo-culting an architecture the data does not
justify.

### Why `AgentExecutor`

It is the smallest LangChain surface that satisfies the tool-calling requirement. It is
legacy-flagged in current LangChain, which steers new work toward LangGraph, and a reviewer may
reasonably ask about that. My answer: LangGraph's checkpointing and graph machinery is ceremony
around one agent loop and one validation gate. The workflow here genuinely is linear. I would
reach for LangGraph when there are branches worth drawing.

### Why manual review reports $0.00

A reviewer scanning the dashboard must never see money in the "approved" column that nobody
authorised. `MANUAL_REVIEW` therefore reports `0.00` for both amounts, and the provisional
figures — what *would* be reimbursable — travel in the explanation and the
`ManualReviewPacket` instead. CLM-004's packet carries $2,800 provisional reimbursable and
$200 provisional deducted; the result object carries zeros. Total approved across the batch is
$1,950, and that number is true.

### Why CLM-002 is REJECT and not MANUAL_REVIEW

Its title states no business purpose, and POL-CAT-01 reimburses only documented business
expenses — which looks like grounds to escalate. But both line items are spa and minibar
charges, flatly ineligible under POL-CAT-02, so there is nothing reimbursable and nothing for a
human to decide. The Decision Guidance defines exactly this case as Reject. The precedence
ladder therefore tests "all lines ineligible" *before* it collects blockers. The missing
business purpose is still recorded as an observation.

### Where the LLM genuinely earns its place

Three things the deterministic engine cannot do:

1. **Deciding what to look at.** Tool order is not fixed; CLM-002 needs almost no evidence,
   CLM-004 needs everything.
2. **Explaining in prose a claimant would accept.** "$200 over the nightly cap because the trip
   dates give two nights, not the three claimed" is a sentence, not a rule firing.
3. **Noticing what the rules do not cover.** `POLICY_SILENT` on CLM-005 is the interesting
   case: the policy caps meals per day and says nothing about four people at a client dinner.
   A rules engine has no way to represent "this rule does not reach this situation".

### What I would do next

* **Calibrate confidence.** The rubric is reasonable and reproducible but uncalibrated — with
  n=5 there is nothing to calibrate against. With a few hundred adjudicated claims I would fit
  the thresholds to observed reviewer agreement.
* **Close the loop.** Reviewer outcomes on escalated claims are the signal that tells you
  whether the escalation rules are too tight or too loose.
* **Policy versioning.** Rules change; decisions must be reproducible against the policy as it
  stood on the claim date.
* **Make the receipt check real.** Right now "receipt attached" is a boolean, so POL-RCT-01's
  *itemized* requirement is unverifiable (see limitations).
""")

# ---------------------------------------------------------------- 16
md(r"""
## 16. Assumptions and limitations

### Assumptions — each one a judgement call that could have gone another way

| # | Assumption | Why | If wrong |
|---|---|---|---|
| 1 | Per-diem units come from **trip dates**, with the description's quantity as a cross-check | The dates are structured and verifiable; the description is free text the claimant wrote | CLM-004's lodging cap becomes $600 instead of $400 — the decision is unchanged, the provisional figure moves |
| 2 | POL-TIME-01 is anchored on **`trip_start`** | Strictest defensible reading: the earliest expense could fall on day one | No outcome changes; all five clear the window by 16+ days either way |
| 3 | The meals cap is **$75 per claimant per day**, with no per-person scaling | POL-PD-01 says "$75 per day" and is silent on headcount; scaling it would be inventing a rule | CLM-005's provisional reimbursable moves from $75 to $300 — it is escalated either way |
| 4 | `MANUAL_REVIEW` reports **$0.00 / $0.00** | Nothing is authorised until a human approves | Dashboard "approved" would include $3,220 nobody has approved |
| 5 | `confidence` means **certainty the decision is right**, escalation included | Uncertainty about money is already carried by the decision field | Every escalation would score low and the field would duplicate the decision |
| 6 | A claim with **all lines ineligible** is REJECT, never MANUAL_REVIEW | The Decision Guidance defines this case explicitly | CLM-002 would escalate instead of rejecting |
| 7 | `receipt_attached = true` implies the receipt is **itemized** | Appendix B gives only a boolean | POL-RCT-01 is partially unenforced — see limitations |
| 8 | Appendix B category labels map onto the POL-CAT bullet lists (`spa` → "Spa, gym and personal entertainment") | Direct lexical correspondence | Unmapped categories hit `UNKNOWN_CATEGORY` and escalate — they fail safe |

### Limitations — known gaps, stated plainly

1. **Receipt itemization is unverifiable.** POL-RCT-01 requires an *itemized* receipt. The data
   supplies only "attached: yes/no". A claim can therefore pass the receipt check with a
   non-itemized receipt, and the pipeline cannot tell. This is why confidence is capped at 0.99
   and never 1.0.
2. **Confidence is uncalibrated.** It is a defensible rubric, not a probability. With n=5 it
   could not be otherwise, and presenting it as calibrated would be dishonest.
3. **No duplicate detection.** The brief lists it as a candidate tool, but no claim history was
   supplied. A duplicate checker that can only ever return zero hits is decoration, so it is
   not built.
4. **n=5.** Every aggregate on the dashboard is computed over five claims. The decision
   breakdown is four counts. Nothing here generalises statistically, and it is not meant to.
5. **Single currency.** The policy specifies USD and every claim is in USD, CLM-004 included
   despite being international. No FX handling exists.
6. **Business purpose is inferred from the claim title** by keyword, which is a crude proxy for
   POL-CAT-01's "documented business purpose". It is recorded as an observation and never
   drives a decision on its own.
7. **The agent is single-turn per claim.** It cannot ask the employee a clarifying question;
   it escalates instead. For a prototype that is the right trade, but a real system would want
   a cheap "request receipt" path that does not consume a reviewer.
8. **`SIMULATED` mode is not a language model.** Without an API key the tool *choices* are a
   fixed plan and the prose is template-generated. The decisions and amounts are identical to a
   live run because they come from the engine — but do not read the simulated run as evidence
   about model behaviour. Set `ANTHROPIC_API_KEY` for that.
""")

# ---------------------------------------------------------------- 17
md(r"""
## 17. Demo evidence

What a reviewer can verify without running anything:

* **Section 10** — all five claims adjudicated, with decision, amounts, confidence and tool
  counts.
* **Section 11** — three complete audit trails: rules retrieved, every tool call with timing,
  the engine ledger line by line, what the gate changed, the final object and the review packet.
* **Dashboard** — four panels over the real outcomes, exported to `UI SS_1.png`; the per-claim
  review card exported to `UI SS_2.png`.
* **Section 13** — ten checks, including the golden table hand-derived from Appendix A.
* **Section 14** — the gate catching a deliberately wrong draft, with the dollar figure it
  prevented.
* **`outputs/`** — `decisions.json` and `audit_trail.json`, written by the cell below and
  committed to the repository so the results are readable without executing anything.
""")
code(r'''
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)

Path("outputs/decisions.json").write_text(
    json.dumps([r.result for r in RUNS], indent=2))
Path("outputs/audit_trail.json").write_text(
    json.dumps([r.audit_record() for r in RUNS], indent=2, default=str))
Path("outputs/evaluation.json").write_text(
    json.dumps(EVAL.frame().to_dict(orient="records"), indent=2))
fig.savefig("outputs/dashboard.png", dpi=150, facecolor="#fcfcfb", bbox_inches="tight")

for p in sorted(Path("outputs").iterdir()):
    print(f"  {p}  ({p.stat().st_size:,} bytes)")
print(f"  UI SS_1.png  ({Path('UI SS_1.png').stat().st_size:,} bytes)")
print(f"  UI SS_2.png  ({Path('UI SS_2.png').stat().st_size:,} bytes)")
''')

# ---------------------------------------------------------------- FINAL
md(r"""
---

## Final output — the required JSON array

One object per Appendix B claim, each with exactly the nine required fields:
`claim_id`, `decision`, `approved_amount`, `deducted_amount`, `missing_docs`, `policy_refs`,
`confidence`, `explanation`, `tools_used`.
""")
code(r'''
FINAL_RESULTS = [r.result for r in RUNS]

assert len(FINAL_RESULTS) == 5, "one object per Appendix B claim"
REQUIRED_FIELDS = {"claim_id", "decision", "approved_amount", "deducted_amount",
                   "missing_docs", "policy_refs", "confidence", "explanation",
                   "tools_used"}
for obj in FINAL_RESULTS:
    assert set(obj) == REQUIRED_FIELDS, f"field mismatch on {obj['claim_id']}"
    assert obj["decision"] in {"APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"}

print(json.dumps(FINAL_RESULTS, indent=2))
''')

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.6",
                      "mimetype": "text/x-python",
                      "file_extension": ".py", "pygments_lexer": "ipython3"},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(OUT))
print(f"wrote {OUT}")
print(f"  {len(cells)} cells  "
      f"({sum(1 for c in cells if c.cell_type == 'code')} code, "
      f"{sum(1 for c in cells if c.cell_type == 'markdown')} markdown)")
