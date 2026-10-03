# Travel Reimbursement Approval Agent — Design Review

**Pre-implementation architecture document**

| | |
|---|---|
| **Deliverable** | `AntrikshVashistha.ipynb` |
| **Policy** | 12 rules, transcribed verbatim from Appendix A |
| **Claims** | 5, transcribed verbatim from Appendix B |
| **Status** | Approved to build |
| **Date** | 2026-10-03 |
| **Shareable page** | https://claude.ai/artifact/9EUHT3fXBjamajbHFXy69W |

An agentic adjudication prototype where a deterministic policy engine owns every dollar and every
hard rule, and the LLM owns retrieval, synthesis, and explanation. The two meet at a gate the LLM
cannot win.

---

## 0. Decisions locked before build

Four choices that would have meant reworking the engine if guessed wrong.

| Decision | Choice | Rationale |
|---|---|---|
| **Manual-review amounts** | Zeros | No money is authorised until a human approves. Provisional figures travel in the explanation and the review packet instead, so dashboard totals never overstate what the agent actually cleared. |
| **Agent framework** | LangChain `AgentExecutor` | `@tool` functions over `ChatAnthropic`, versions pinned. Offline replay needs a `BaseChatModel` subclass so one code path serves both modes. |
| **Keyless execution** | Cached transcript | Recorded from a real live run and committed. Every cell, chart and result renders without an API key, behind a visible replay banner. |
| **Per-diem unit source** | Trip dates authoritative | Derived counts cross-checked against each line's stated quantity. Divergence raises `DATA_CONFLICT` rather than resolving silently. |

---

## 1. Proposed architecture

Seven thin layers. The load-bearing idea is a hard split between the money and the language.

```
┌─ L0  DATA (transcribed verbatim from Appendix A & B) ──────────────┐
│  PolicyRegistry: 12 typed PolicyRule objects (id, text, params)    │
│  ClaimSet: 5 Pydantic Claims / 12 LineItems                        │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ L1  DETERMINISTIC POLICY ENGINE (pure Python, Decimal) ───────────┐  ◀── authority
│  Sole authority for: eligibility, caps, day/night counts, totals,  │
│  tiers, timeliness, conflicts.  Emits an immutable RuleLedger.     │
└────────────────────────────────────────────────────────────────────┘
                              │ engine fns wrapped as tools
┌─ L2  TOOL SURFACE (6 LangChain tools + schemas + call audit) ──────┐
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ L3  AGENT LOOP (AgentExecutor + ChatAnthropic) ───────────────────┐
│  Chooses tools, handles tool errors, synthesizes, drafts decision  │
└────────────────────────────────────────────────────────────────────┘
                              │ draft (untrusted)
┌─ L4  RECONCILIATION GATE ──────────────────────────────────────────┐  ◀── authority
│  Pydantic schema validation → amounts/decision overwritten from    │
│  ledger → LLM may only ESCALATE to MANUAL_REVIEW, never relax →    │
│  policy_refs grounded against registry (hallucinated IDs dropped)  │
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ L5  CONFIDENCE SCORER (deterministic rubric) ─────────────────────┐
└────────────────────────────────────────────────────────────────────┘
                              │
┌─ L6  OUTPUT + UI/DASHBOARD      ┌─ L7  EVAL HARNESS ───────────────┐
│  JSON array, pandas, matplotlib │  golden table, determinism,      │
│  ipywidgets panel, UI SS_*.png  │  grounding, ablation             │
└─────────────────────────────────┴──────────────────────────────────┘
```

**Why engine-authoritative rather than LLM-authoritative-with-checks.** The only way to *guarantee*
the LLM is not the source of truth for arithmetic — rather than hope for it — is to compute the
ledger before the agent runs and overwrite the agent's numbers after. The agent still does real
agentic work: it decides which tools to call and in what order, and its escalation judgement is
honoured. It simply cannot emit a wrong dollar figure, by construction.

---

## 2. Component responsibilities

Each component's boundary is defined as much by what it refuses to own as by what it does.

| Component | Owns | Does NOT own |
|---|---|---|
| `PolicyRegistry` | Verbatim rule text + machine params (`meals_per_day=75`), lookup by id/category/keyword | Interpretation, precedence |
| `Claim` / `LineItem` | Input shape, type coercion, field validation | Business rules |
| `PolicyEngine` | All arithmetic, all hard-rule verdicts, conflict detection, the precedence ladder | Natural language, tone |
| `RuleLedger` | Immutable per-line and per-claim findings, each tagged with its `POL-*` id | — |
| Tool layer | Schemas, dispatch, error-to-LLM translation, call audit | Deciding *when* to call |
| Agent loop | Tool selection, synthesis, draft decision, explanation | Final numbers, final decision |
| `ReconciliationGate` | Schema conformance, engine-wins override, escalation-only rule, reference grounding | Recomputing policy |
| `ConfidenceScorer` | Rubric → float, low-confidence escalation guard | — |
| UI / dashboard | Claim selector, run control, result render, 4 charts from real outcomes | Any business logic |
| Eval harness | Golden assertions, determinism, grounding, ablation | — |

---

## 3. Agent workflow

One agent session per claim. Seven steps, of which only step 3 is non-deterministic.

1. **Pre-flight.** The engine computes the full `RuleLedger` before the LLM is invoked. This is
   ground truth, held aside.
2. **Prompt.** System prompt carries the role, the four enums, the precedence ladder, the nine-field
   contract, and the hard instruction never to compute money. The user message is the claim JSON
   alone — *the policy text is deliberately withheld*, so the agent must retrieve it. That is what
   makes grounding demonstrable rather than asserted.
3. **Agentic loop** (max 6 turns). Traces differ genuinely by claim: `CLM-002` short-circuits after
   eligibility; `CLM-004` runs policy lookup → receipt check → limits → conflict detection →
   threshold.
4. **Missing and conflicting information.** Tools return structured findings with
   `severity ∈ {info, deduction, blocker, conflict}` rather than throwing. Every blocker or conflict
   must be addressed in the explanation, or the gate escalates on the agent's behalf.
5. **Draft.** Emitted through a `submit_decision` tool call, so the output arrives schema-shaped
   rather than as JSON wrapped in prose.
6. **Gate.** Reconcile, score confidence, finalise the `ClaimDecision`.
7. **Audit record.** Retrieved rule ids, every tool call with arguments and results, token counts,
   the raw LLM draft, the diff the gate applied, and the final object.

**Fallbacks.** A malformed draft gets one repair turn carrying the validation error; still bad, and
it becomes manual review at `confidence 0.0` with reason code `AGENT_OUTPUT_INVALID`. An API error
retries with backoff, then falls back to an engine-only decision flagged `LLM_UNAVAILABLE`. The
pipeline never crashes and never silently approves.

---

## 4. Proposed tools

Six, against a stated minimum of two. Each is a real engine function, each cites policy, each is
independently testable.

| Tool | Returns | Cites |
|---|---|---|
| `policy_lookup` | Matching rules: id, verbatim text, type, parameters | — |
| `check_receipt_completeness` | Per-line required / attached / ok, plus `missing_docs` | POL-RCT-01, POL-RCT-02 |
| `compute_limits` | Per-line eligible / allowed / excess; day and night derivation; reimbursable and deducted totals | POL-CAT-01/02, POL-PD-01/02/03, POL-AIR-01 |
| `check_approval_threshold` | Tier, and whether the total sits within agent authority | POL-APR-01/02/03 |
| `check_timeliness` | Days elapsed, within-window flag | POL-TIME-01 |
| `detect_conflicts` | Line-sum vs stated total, stated vs derived units, unknown category, undocumented business purpose, per-diem headcount ambiguity | Decision Guidance |

Retrieval is keyword and category matching over twelve typed records. **No vector store.** With a
one-page policy, embeddings would buy fuzzy-match failure modes, non-determinism and a dependency in
exchange for nothing — stated plainly in the Design Notes, because *why no RAG?* is a certainty at
review.

---

## 5. Structured output design

Exactly nine fields, no extras, enforced by `ConfigDict(extra="forbid")`.

```python
claim_id:        str                      # "CLM-001"
decision:        Literal["APPROVE","PARTIAL_APPROVE","REJECT","MANUAL_REVIEW"]
approved_amount: float                    # Decimal internally, 2dp on serialize
deducted_amount: float
missing_docs:    list[str]                # [] when none; human-actionable
policy_refs:     list[str]                # validated POL-* ids, deduped, ordered
confidence:      float                    # 0.0-1.0, 2dp
explanation:     str                      # 2-5 sentences, each claim traceable
tools_used:      list[str]                # actual calls, in order
```

### Invariants the gate enforces

- `APPROVE` implies `deducted_amount == 0`; `REJECT` implies `approved_amount == 0`;
  `PARTIAL_APPROVE` implies both are positive.
- A non-empty `missing_docs` implies the decision is `MANUAL_REVIEW`.
- Every entry in `policy_refs` resolves to a rule in the registry.

The full audit trail lives in a *separate parallel object*, never in these nine fields. The contract
says exactly, and that is honoured literally.

---

## 6. Validation strategy

Five independent gates, and one precedence ladder run in fixed order.

1. **Input.** Pydantic on claims: enum categories, non-negative amounts, `trip_start ≤ trip_end`,
   dates parse.
2. **Arithmetic integrity.** Sum of line amounts against the claim's stated total. All five
   reconcile — it is a guard, not a formality.
3. **Deterministic rule evaluation.** The ladder below.
4. **Output.** Pydantic, plus the invariants, plus reference grounding.
5. **Reconciliation.** Engine amounts and decision overwrite the draft. Every override is logged and
   surfaced, so a reviewer can see how often engine and model agreed — a real signal, and good demo
   evidence.

### Precedence ladder

```
1. classify each line            POL-CAT-01 / POL-CAT-02 / unknown
2. ALL lines ineligible       → REJECT          deduct in full
3. collect blockers:             non-economy airfare (AIR-01)
                                 missing required receipt (RCT-01/02)
                                 late submission (TIME-01)
                                 total > $2,000 (APR-03)
                                 any conflict · unknown category
                                 confidence below floor
4. any blocker                → MANUAL_REVIEW
5. deductions > 0             → PARTIAL_APPROVE
6. otherwise                  → APPROVE          tier APR-01 or APR-02
```

Reject precedes manual review because the Decision Guidance defines rejection as *ineligible with
nothing reimbursable* — there is nothing for a human to review. That ordering is documented, not
buried.

> **Determinism trap, guarded explicitly.** Nothing may call `datetime.now()`. Today is October 2026
> and the claims are from June 2026 — a naive `now()` marks all five late and silently corrupts every
> result. Timeliness is computed strictly from `submitted_date` against the expense date, and the
> notebook asserts the clock is frozen.

---

## 7. Manual-review strategy

A first-class outcome, not a fallback. Each routed claim carries a `ManualReviewPacket`: reason
codes, what the human must check, what to request from the employee, and the provisional figures the
reviewer starts from.

| Reason code | Trigger | Rule |
|---|---|---|
| `POLICY_EXCEPTION_AIRFARE_CLASS` | Business or first class present | POL-AIR-01 |
| `MISSING_REQUIRED_RECEIPT` | Receipt required, not attached | POL-RCT-01/02 |
| `EXCEEDS_AGENT_AUTHORITY` | Reimbursable above $2,000 | POL-APR-03 |
| `LATE_SUBMISSION` | Beyond the 30-day window | POL-TIME-01 |
| `DATA_CONFLICT` | Stated vs derived units, or line-sum mismatch | Decision Guidance |
| `POLICY_SILENT` | Scenario the policy does not cover | Decision Guidance |
| `UNKNOWN_CATEGORY` | Category in neither CAT-01 nor CAT-02 | POL-CAT-01/02 |
| `LOW_CONFIDENCE` | Score below the floor | — |

Escalation is **monotonic**: any layer may escalate, no layer may de-escalate. That single property
is what makes *prefer manual review over forcing a decision* structurally true rather than
prompt-dependent.

---

## 8. Confidence strategy

A deterministic rubric computed in the gate — not asked of the LLM as source of truth, since
self-reported confidence is poorly calibrated and would be irreproducible.

Base score by decision cleanliness, then additive penalties: each missing document −0.10, each
conflict −0.15, `POLICY_SILENT` −0.20, engine-model disagreement −0.15, output repair −0.20,
`LLM_UNAVAILABLE` −0.25. Clamped to `[0.0, 0.99]` — never 1.0, because receipt itemisation is
unobservable from the supplied data.

The model's self-reported confidence *is* captured, and may only lower the final score — a cheap,
honest way to use a weak signal without trusting it. Divergence between the two is charted.

> **A semantic fork worth naming.** Two readings of `confidence` produce opposite numbers. Either it
> means certainty that the returned `decision` is correct — so a clear-cut manual review scores
> *high* — or certainty that automated adjudication is safe, so manual review scores *low*.
>
> This build takes the first. Uncertainty about the money is already expressed by routing to manual
> review; double-encoding it as low confidence makes high-confidence escalation inexpressible.
> `CLM-004` lands near 0.92, `CLM-005` near 0.72, since policy silence is genuinely murkier than a
> plain missing receipt. The alternative reading is stated in the Design Notes so the choice reads as
> deliberate.

---

## 9. Evaluation strategy

| Test | Method | Pass bar |
|---|---|---|
| Golden outcomes | Expected decision and amounts for all five, hand-derived from Appendix A | 5/5 exact |
| Schema conformance | All five validate; exactly nine keys | 5/5 |
| Reference grounding | Every emitted `POL-*` exists in the registry | 0 hallucinated |
| Engine determinism | Engine ×20, byte-compared | identical |
| Pipeline stability | Full agent ×3; decision and amounts compared | identical |
| Enum coverage | All four decisions exercised by the supplied set | 4/4 |
| Ablation | LLM-only, no tools or engine, same prompts, diffed against the full pipeline | shows arithmetic drift |
| Robustness | Deliberately corrupt fixture: null amount, unknown category | graceful escalation, no crash |
| Tool exercise | Every tool called at least once across the batch | 6/6 |

The ablation is the single most persuasive cell in the notebook: it demonstrates *why* the
architecture is shaped this way instead of asserting it. The robustness fixture is labelled as a test
fixture and excluded from the five required results.

---

## 10. The supplied data, and the outcomes it implies

Appendix B covers all four decisions — manual review twice, for two unrelated reasons. Clearly
deliberate on the examiner's part, and the golden table is where that reading gets proven.

| Claim | Decision | Claimed | Approved | Deducted | Driving rules |
|---|---|---:|---:|---:|---|
| CLM-001 | APPROVE | 1,110.00 | 1,110.00 | 0.00 | Fully compliant; POL-APR-02 tier |
| CLM-002 | REJECT | 380.00 | 0.00 | 380.00 | Spa and minibar, both POL-CAT-02; no documented business purpose |
| CLM-003 | PARTIAL_APPROVE | 940.00 | 840.00 | 100.00 | Lodging $250/night against a $200 cap, POL-PD-02 |
| CLM-004 | MANUAL_REVIEW | 3,000.00 | 0.00 | 0.00 | Business class POL-AIR-01; missing lodging receipt; above $2,000; nights conflict |
| CLM-005 | MANUAL_REVIEW | 220.00 | 0.00 | 0.00 | Missing receipt POL-RCT-01; policy silent on a four-person client dinner |
| **Totals** | | **5,650.00** | **1,950.00** | **480.00** | $3,220 awaiting human judgement |

### Two planted problems, neither papered over

> **CLM-004 · a genuine internal conflict.** The trip runs 2026-06-16 to 2026-06-18, which is **two
> nights**. The line item reads **three nights** for $600. At the $200 cap, two nights allows $400 and
> leaves $200 excess; three nights allows the full $600. The decision does not move — business class,
> a missing receipt and a $3,000 total each force escalation independently — but the provisional
> figure does, and so does whether a conflict gets reported at all. Trip dates win; the divergence is
> raised as `DATA_CONFLICT`.

> **CLM-005 · policy silence.** A client dinner for four at $220 on a single day. POL-PD-01 caps
> meals at $75 per day and says nothing about headcount or client entertainment. Both $75 and $300
> are arguable, so the cap stays at $75 for the claimant — the policy authorises no per-person
> scaling — and `POLICY_SILENT` is attached. The packet asks the reviewer to confirm whether client
> entertainment is in scope at all.

All five claims clear POL-TIME-01 comfortably, at eight to fourteen days, so timeliness moves no
outcome. It is implemented regardless.

---

## 11. Notebook structure

Eighteen sections, top to bottom, no manual steps. The brief requires the final *code* cell to emit
the JSON, so all prose sits above it and nothing follows.

| § | Content |
|---|---|
| 0 | Title and README — setup, environment variables, how to run, design choices, sixty-second tour |
| 1 | Dependencies, pinned, idempotent install; Python version note |
| 2 | Config — `ANTHROPIC_API_KEY`, `MODEL`, `LIVE_LLM` toggle, frozen-clock assertion |
| 3 | Appendix A as code — 12 rules, verbatim text plus parameters, provenance per rule |
| 4 | Appendix B as data — 5 claims, 12 line items |
| 5 | Output contract — `ClaimDecision`, nine fields, invariants |
| 6 | Deterministic engine, `RuleLedger`, precedence ladder |
| 7 | Tool layer — six schemas, implementations, audit log |
| 8 | Agent loop — LangChain executor, retries, fallbacks |
| 9 | Reconciliation gate and confidence scorer |
| 10 | Orchestrator — run all five, assemble audit trail |
| 11 | Sample outputs — three full walkthroughs with retrieved context, tool trace and gate diff |
| 12 | **## Dashboard** — interactive panel, four charts, exports `UI SS_1.png` and `UI SS_2.png` |
| 13 | Evaluation harness and results table |
| 14 | Ablation demo — LLM-only against the full pipeline |
| 15 | **Design Notes & Reasoning** |
| 16 | Assumptions and limitations |
| 17 | Demo evidence |
| final code cell | `print(json.dumps(results, indent=2))` — the array of five |

---

## 12. Repository structure

Flat and boring on purpose.

```
travel-reimbursement-approval-agent/
├── AntrikshVashistha.ipynb      ← the only assessed deliverable; self-contained
├── DESIGN.md                    ← this document
├── README.md                    ← GitHub landing page, mirrors notebook §0
├── requirements.txt             ← pinned
├── .env.example                 ← ANTHROPIC_API_KEY=sk-ant-...
├── .gitignore
├── UI SS_1.png                  ← dashboard; the brief names this file
├── UI SS_2.png                  ← interactive panel
└── outputs/
    ├── decisions.json           ← committed, so results are visible without running
    ├── audit_trail.json
    └── dashboard.png
```

No `src/` package. The brief wants one runnable notebook, and lifting logic into importable modules
would mean the notebook no longer demonstrates the implementation. The notebook *writes* `outputs/`
but never reads it, so a fresh clone runs identically. The brief PDF is not committed — it may be the
examiner's proprietary material, and Appendices A and B appear as transcribed data with provenance
instead.

---

## 13. Deliberately not built

Each omission is defensible out loud, which is worth more than the feature.

- **Vector DB, embeddings, RAG** — twelve rules, roughly 800 words. A typed registry with keyword
  lookup is exact, auditable and instant.
- **Multi-agent crews** — more non-determinism, more tokens, no new capability. One agent with good
  tools beats five agents negotiating.
- **Any database, SQLite included** — five claims live in memory.
- **FastAPI, microservices, Docker, Kubernetes** — the deliverable is a notebook.
- **OCR and receipt image parsing** — Appendix B supplies a boolean, not images. Parsing nothing
  would be theatre.
- **Fine-tuning** — twelve rules fit in a prompt.
- **Currency conversion** — the policy says USD; CLM-004 is international but priced in USD.
- **Auth, RBAC, workflow engine, notifications** — manual review emits a packet; routing it is out of
  scope.
- **LangGraph** — graph ceremony around one agent loop and one gate.
- **Self-consistency voting** — determinism comes from the engine, not from re-rolling the model.
- **Duplicate-history store** — no history supplied, and inventing one would breach the constraints.
- **A web frontend** — the UI requirement is met in-notebook.

---

## 14. Risks and likely review questions

| Risk | Mitigation |
|---|---|
| Model contradicts the engine on money | Structurally impossible — the gate overwrites |
| Model cites a plausible but fake `POL-*` id | References validated against the registry; unknowns dropped and logged |
| Notebook will not run for the reviewer | Cached-transcript offline mode |
| "Agentic" reads as a thin wrapper | Policy withheld from the prompt forces retrieval; traces differ per claim; the ablation proves the LLM-only path fails |
| Over-escalation — everything becomes manual review | Golden table asserts CLM-001 approve and CLM-003 partial, so escalation cannot quietly swallow them |
| `now()` creeping in and marking all claims late | Frozen-clock assertion |
| `ipywidgets` not rendering on GitHub | Static matplotlib dashboard plus committed PNGs as the durable artifact |
| Float arithmetic on money | `Decimal` throughout; 2dp only at serialisation |
| Prompt injection through claim description text | Descriptions are data; the engine never reads them for authority; the agent prompt treats them as untrusted |

### Questions expected at review, and where the notebook answers them

1. Why no vector database? — §15
2. What stops the model getting the arithmetic wrong? — §9 gate, §14 ablation
3. Is the LLM doing anything real, or is this a rules engine with a chatbot bolted on? — §11 traces,
   §14 ablation
4. Why is CLM-002 a reject rather than manual review, given no documented business purpose? — the
   precedence ladder, §6
5. For manual review, what do the amounts mean? — zeros, by the decision recorded above
6. Where does confidence come from, and is it calibrated? — §9; the honest answer is a rubric,
   uncalibrated, n=5
7. What happens when the API is down? — §8 fallback
8. How would this scale to ten thousand claims and a 200-page policy? — §15, and this is precisely
   where retrieval would earn its place
9. CLM-004 says three nights but the dates imply two — which did you use? — trip dates, with the
   conflict reported
10. Is per-diem per person or per claim? — per claimant; the policy authorises no scaling
11. How do you know a receipt is *itemised*, as POL-RCT-01 requires? — §16; you do not, and it is
    unobservable
12. What is the test coverage? — §13

> **One caveat carried into the build.** `AgentExecutor` is legacy-flagged in current LangChain, which
> steers new work toward LangGraph. It still works and is widely recognised, so it is a reasonable
> call for an interview deliverable — but a sharp reviewer may ask why a deprecated API. The answer
> going into the Design Notes: it is the smallest LangChain surface that satisfies the tool-calling
> requirement, and LangGraph's machinery is unjustified ceremony around one agent and one validation
> gate.

---

## 15. Resolved defaults

Lower-stakes calls, settled rather than assumed, each stated so it can be overridden.

| Question | Resolution |
|---|---|
| Confidence semantics | Certainty the `decision` is correct, so clear-cut escalation scores high. Alternative reading documented. |
| Per-diem headcount | $75 per day for the claimant; no per-person scaling. `POLICY_SILENT` attached to CLM-005. |
| Timeliness anchor | `trip_start` — the strictest defensible reading. No outcome moves either way. |
| UI shape | `ipywidgets` console degrading to a plain function call, plus a four-panel static dashboard and two exported PNGs. |
| Robustness fixture | Built, loudly labelled, confined to §13, excluded from the five results. |
| `detect_duplicates` | Skipped. No claim history supplied; a checker that can only ever return zero hits is decoration. |
| Brief PDF in the repo | Not committed. Appendices transcribed with provenance instead. |
| Model | `claude-sonnet-5` — strong tool use, fast, cheap enough to re-run the batch freely. Configurable at the top. |
| Data location | Embedded inline, so a fresh clone runs with zero file dependencies. JSON copies written as a convenience, never read. |
| Python and money types | Virtual environment on 3.11+, asserted at ≥3.10. `Decimal` internally, 2dp at serialisation. |


---

# Post-review amendments

This document records the design **as approved before implementation**. It is left intact for
provenance. A critical audit of the built system found defects that changed the following;
the notebook and README describe the current behaviour.

| Finding | Change |
|---|---|
| **C1** The gate validated eight of the nine fields and passed `explanation` through on a length check, so prose justifying an *overridden* verdict shipped beside the corrected one | `validate_explanation()` rejects prose that argues for a different decision, invents a figure, cites an unknown rule, or exceeds five sentences; rejected prose falls back to the deterministic explanation and costs 0.10 confidence |
| **C2** `units <= 0 -> 1` granted a full night's lodging cap on a zero-night trip | The floor applies to day-based per-diems only; a lodging line on a zero-night trip raises `DATA_CONFLICT` |
| **C3** `\bfor\s+(\d+)\b` read "Hotel for 3 nights" as three people, producing false escalations | Headcount requires an explicit people-noun, or a bare "for N" not followed by a unit noun |
| **C4** Airfare with no stated class defaulted to economy and could auto-approve | Unstated and "premium economy" both raise `AIRFARE_CLASS_UNVERIFIED` and escalate |
| **M2** `policy_refs` cited every rule checked, so POL-TIME-01 and POL-CAT-01 appeared on all five claims | Only rules that changed an amount, blocked the claim, or authorised the approval are cited |
| **H1** The documented repair turn did not exist | One real repair turn: the validation error is returned to the agent, which tries again |
| **H2** Failure kinds were inferred by substring-matching an error message, so a schema failure was treated as an API outage | Typed `failure_kind`; `AgentDraft` tightened so the invalid path is reachable at all |
| **L1** The build sources lived outside the repository | Moved to `notebook_src/` with its own README |

Two confidence changes follow from the above: rejected prose costs 0.10, and
`AIRFARE_CLASS_UNVERIFIED` joins the unambiguous blockers. The five Appendix B outcomes are
unchanged: $1,950 approved, $480 deducted.

**Headline finding, addressed in a second pass.** The audit's most serious structural point was
that the `SIMULATED` stand-in composed its submission by calling `ENGINE.adjudicate()` directly,
so every headline metric — the golden table, determinism, pipeline stability — was the engine
agreeing with itself. The stand-in now builds its draft from the tool JSON it receives and never
touches the engine, and two checks make that provable: **Draft is tool-derived** feeds it a
doctored `compute_limits` payload and asserts the draft follows the wrong data while the engine
holds and the gate flags the disagreement; **Agent/engine agreement** asserts all five drafts
were actually produced and matched the engine independently, failing if the agent crashed or
returned nothing. Fixing this surfaced a latent bug of its own: a leading-underscore class
attribute on a pydantic model became a `ModelPrivateAttr`, which crashed the stand-in on the two
escalated claims — caught safe by the gate, but silently, which is why the agreement check now
fails on any agent failure.

**Not addressed in this pass** (known, deliberate): `REASON_RULES` remains dead code; `stated_rate` is parsed and
displayed but never consumed; the engine recomputes per claim; `ReconciliationGate` is named as a
component here but implemented as the function `reconcile`.
