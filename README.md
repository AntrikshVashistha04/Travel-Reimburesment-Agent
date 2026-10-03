# Travel Reimbursement Approval Agent

A GenAI / Agentic AI prototype that adjudicates employee travel reimbursement claims against a
supplied policy and returns a structured recommendation for each claim.

**The deliverable is a single notebook: [`AntrikshVashistha.ipynb`](AntrikshVashistha.ipynb).**
It runs top to bottom with no manual steps and **no API key required**.

---

## The one idea this is built on

> **A deterministic policy engine owns every dollar and every hard rule.
> The LLM owns retrieval, tool selection, synthesis and explanation.
> They meet at a reconciliation gate that the LLM cannot win.**

The agent does real agentic work — it is handed the claim **but not the policy**, so it must
retrieve the rules itself, decide which checks to run, combine their results and resolve
conflicts. What it cannot do is produce a wrong number: the ledger is computed before the agent
runs, and the gate overwrites the agent's arithmetic afterwards.

Escalation is **monotonic**: any layer may route a claim to `MANUAL_REVIEW`, no layer may relax
one. That is what makes "prefer Manual Review over forcing a decision" structurally true rather
than dependent on prompt wording.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate     # Python 3.9+ (3.11+ recommended)
pip install -r requirements.txt
jupyter lab AntrikshVashistha.ipynb                     # then Run All
```

To run the agent against the real Claude API, set a key first (optional):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Results

| Claim | Decision | Claimed | Approved | Deducted | Driving rules |
|---|---|---:|---:|---:|---|
| CLM-001 | `APPROVE` | 1,110.00 | 1,110.00 | 0.00 | Fully compliant; POL-APR-02 tier |
| CLM-002 | `REJECT` | 380.00 | 0.00 | 380.00 | Spa + minibar, both POL-CAT-02 |
| CLM-003 | `PARTIAL_APPROVE` | 940.00 | 840.00 | 100.00 | Lodging $250/night vs $200 cap, POL-PD-02 |
| CLM-004 | `MANUAL_REVIEW` | 3,000.00 | 0.00 | 0.00 | Business class, missing receipt, >$2,000, nights conflict |
| CLM-005 | `MANUAL_REVIEW` | 220.00 | 0.00 | 0.00 | Missing receipt; policy silent on a 4-person client dinner |
| **Total** | | **5,650.00** | **1,950.00** | **480.00** | $3,220 awaiting human judgement |

`MANUAL_REVIEW` reports `0.00` for both amounts by design — nothing is authorised until a human
approves. The provisional figures travel in the explanation and the manual-review packet, so the
dashboard's "approved" total never overstates what the agent actually cleared.

Machine-readable results are committed in [`outputs/decisions.json`](outputs/decisions.json), so
they can be read without executing anything.

![Dashboard](UI%20SS_1.png)

---

## Execution modes

| Mode | When | What drives tool selection |
|---|---|---|
| `LIVE` | `ANTHROPIC_API_KEY` is set | Claude, via LangChain `AgentExecutor` |
| `REPLAY` | no key, `transcripts.json` present | AI turns recorded from a previous live run |
| `SIMULATED` | no key, no transcript | A deterministic stand-in — **not a language model and not a recording of one** |

**The five results are identical in all three modes**, because every decision and amount comes
from the deterministic engine. What changes is only *who chose which tools to call* and *who
wrote the prose*. The active mode is printed in the notebook banner and carried on every result
card.

The stand-in builds its submission **from the tool results it receives** — it never reads the
engine. A keyless run therefore exercises the tool surface, the precedence ladder and the gate
for real: a test feeds the agent a doctored `compute_limits` payload and asserts the draft
follows the (wrong) tool output while the engine holds its ground and the gate records the
disagreement. That is what stops the golden table being the engine agreeing with itself. What a
keyless run does *not* show is how a real model chooses tools or words an explanation.

---

## Architecture

```
L0  DATA            12 typed PolicyRule records (Appendix A, verbatim)
                    5 Pydantic Claims / 12 line items (Appendix B, verbatim)
         |
L1  ENGINE  <-----  authority: eligibility, caps, unit derivation, totals,
                    tiers, timeliness, conflicts. Decimal throughout.
         |          Emits an immutable RuleLedger.
L2  TOOLS           6 LangChain tools + submit_decision, with a call audit
         |
L3  AGENT           AgentExecutor + ChatAnthropic. Chooses tools, synthesises,
         |          drafts a decision and an explanation.
L4  GATE    <-----  authority: schema validation, amounts and decision
         |          overwritten from the ledger, escalation-only rule,
         |          policy refs grounded against the registry
L5  CONFIDENCE      deterministic rubric; the model may only lower it
         |
L6  UI              JSON array, dashboard, review console, exported PNGs
L7  EVAL            golden table, determinism, grounding, ablation
```

### The six tools

| Tool | Returns | Cites |
|---|---|---|
| `policy_lookup` | Verbatim rules, params, decision guidance | — |
| `compute_limits` | Per-line eligible / allowed / excess, unit derivation, totals | POL-CAT-01/02, POL-PD-01/02/03, POL-AIR-01 |
| `check_receipt_completeness` | Per-line receipt verdicts, `missing_docs` | POL-RCT-01/02 |
| `check_approval_threshold` | Tier and agent authority | POL-APR-01/02/03 |
| `check_timeliness` | Days elapsed, within-window flag | POL-TIME-01 |
| `detect_conflicts` | Unreconciled totals, stated-vs-derived units, unknown categories, policy silence | Decision Guidance |

---

## Key design decisions

- **No vector database.** Twelve rules, ~800 words. A typed registry with keyword lookup is
  exact, auditable and instant. Embeddings would add a dependency and a silent-wrong-rule
  failure mode for nothing. At a 200-page policy this inverts.
- **The policy is withheld from the prompt.** The agent must call `policy_lookup`, which makes
  grounding demonstrable rather than merely asserted.
- **Trip dates are authoritative for per-diem units.** CLM-004 says "Hotel, 3 nights" but its
  dates span two — a real contradiction in the supplied data, raised as `DATA_CONFLICT` rather
  than silently resolved.
- **`confidence` means certainty the decision is correct**, escalation included — so a clear-cut
  `MANUAL_REVIEW` scores *high* (CLM-004: 0.92). Interpretive blockers lower it (CLM-005: 0.72).
- **Frozen clock.** Timeliness uses each claim's own dates. A stray `datetime.now()` would mark
  all five June-2026 claims late; a test asserts the elapsed-day counts.
- **The explanation is validated, not passed through.** It is the field a human reads, so the
  gate rejects prose that argues for a decision other than the one shipped, invents a dollar
  figure, cites a non-existent rule, or runs past five sentences. Rejected prose falls back to
  the deterministic explanation.
- **`policy_refs` carries rules that *drove* the decision**, not every rule that was read. A
  rule earns its place by changing an amount, blocking the claim, or authorising the approval —
  so POL-TIME-01 is not cited on a claim that was comfortably on time.
- **Ambiguity escalates rather than defaulting.** An airfare line that does not state its class,
  or states "premium economy", cannot be approved on the strength of POL-AIR-01 — a reviewer
  confirms the class. A lodging line on a zero-night trip is a contradiction, not a free night.

Full reasoning, including what was deliberately **not** built: [`DESIGN.md`](DESIGN.md), and
section 15 of the notebook.

---

## Verification

Thirteen checks run in section 13; all pass.

| Test | Bar | Result |
|---|---|---|
| Golden outcomes (hand-derived from Appendix A) | 5/5 exact | PASS |
| Schema conformance (exactly 9 fields) | 5/5 | PASS |
| Policy-ref grounding | 0 hallucinated | PASS |
| Engine determinism | identical ×20 | PASS |
| Pipeline stability | identical ×3 | PASS |
| Decision enum coverage | 4/4 | PASS |
| Tool exercise | 7/7 called | PASS |
| Robustness fixture | escalates, no crash | PASS |
| Input validation | 2/2 rejected | PASS |
| Frozen clock | no wall-clock leak | PASS |
| Fallback explanation | valid on all claims | PASS |
| Draft is tool-derived | follows doctored tools | PASS |
| Agent/engine agreement | 5/5 independently | PASS |

Section 14 pushes a deliberately wrong draft through the gate — one that approves CLM-004 in
full, reports no missing documents and cites two invented policy ids. The gate overrides it to
`MANUAL_REVIEW`, **preventing $3,000 of unauthorised approval** and dropping both fake
references.

---

## Repository layout

```
AntrikshVashistha.ipynb   the deliverable; self-contained, runs top-to-bottom
notebook_src/             the modules the notebook is GENERATED from (see its README)
DESIGN.md                 architecture and reasoning, written before implementation
README.md                 this file
requirements.txt          pinned dependencies
.env.example              environment variables
UI SS_1.png               dashboard
UI SS_2.png               single-claim review card
outputs/                  decisions.json, audit_trail.json, evaluation.json, dashboard.png
```

The notebook imports nothing from `notebook_src/` and a reviewer never needs it to run the
deliverable — it exists so the notebook can be *maintained*. Each module was built and tested
standalone before assembly, which is how the engine was verified against the golden table
independently of the notebook plumbing. Edit the modules and rebuild; edits made directly in
the notebook are lost. The notebook *writes* `outputs/` but never reads it, so a fresh clone
runs identically.

---

## Known limitations

1. **Receipt itemization is unverifiable** — the data supplies only a boolean, so POL-RCT-01's
   *itemized* requirement is partially unenforced. This is why confidence caps at 0.99.
2. **Confidence is uncalibrated** — a defensible rubric, not a probability. With n=5 it could
   not be otherwise.
3. **No duplicate detection** — no claim history was supplied; a checker that can only return
   zero hits is decoration.
4. **n=5** — nothing here generalises statistically.
5. **Single currency** — the policy specifies USD.
6. **A `SIMULATED` run tests the pipeline, not the model.** The stand-in derives its draft from
   tool output, so the tools, the ladder and the gate are genuinely exercised — but its tool
   *choices* follow a fixed plan and its prose is templated. Nothing in a keyless run is
   evidence about how an LLM behaves. Set `ANTHROPIC_API_KEY` for that.

The full list is in section 16 of the notebook.
