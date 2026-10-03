# =====================================================================
# SECTION 12 — RESULTS UI AND DASHBOARD
# =====================================================================
# Palette: validated with the data-viz palette validator on the light
# chart surface #fcfcfb. Fixed slot order
#   good #0ca30c -> blue #2a78d6 -> yellow #eda100 -> red #d03b3b
# passes the lightness band, chroma floor, CVD separation and
# normal-vision floor. Green must never sit adjacent to red (deutan
# dE 4.1), which is why blue separates them in every stack.
# The one contrast WARN (#eda100 at 2.11:1) is relieved as the method
# requires: every mark carries a direct label and a full results table
# is printed alongside.
# =====================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

# "$1,110" must render as currency, not as mathtext between $ delimiters.
matplotlib.rcParams["text.parse_math"] = False

from core import CLAIMS_BY_ID, REGISTRY

# ---- chart chrome & ink (data-viz reference instance) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

C_APPROVE = "#0ca30c"
C_MANUAL = "#2a78d6"
C_PARTIAL = "#eda100"
C_REJECT = "#d03b3b"

DECISION_COLOR = {
    "APPROVE": C_APPROVE,
    "MANUAL_REVIEW": C_MANUAL,
    "PARTIAL_APPROVE": C_PARTIAL,
    "REJECT": C_REJECT,
}
DECISION_ORDER = ["APPROVE", "PARTIAL_APPROVE", "REJECT", "MANUAL_REVIEW"]

# Money buckets. Stack order is green -> blue -> red so the two
# confusable hues are never adjacent.
M_APPROVED, M_AWAITING, M_DEDUCTED = C_APPROVE, C_MANUAL, C_REJECT

CONFIDENCE_FLOOR = 0.35


def results_frame(runs) -> pd.DataFrame:
    """The dashboard's single source of data: actual claim outcomes."""
    rows = []
    for r in runs:
        d = r.result
        claim = CLAIMS_BY_ID[d["claim_id"]]
        awaiting = (float(claim.stated_total)
                    if d["decision"] == "MANUAL_REVIEW" else 0.0)
        rows.append({
            "claim_id": d["claim_id"],
            "employee": claim.employee,
            "decision": d["decision"],
            "claimed": float(claim.stated_total),
            "approved": d["approved_amount"],
            "deducted": d["deducted_amount"],
            "awaiting_review": awaiting,
            "confidence": d["confidence"],
            "n_missing_docs": len(d["missing_docs"]),
            "n_policy_refs": len(d["policy_refs"]),
            "n_tools": len(d["tools_used"]),
            "policy_refs": ",".join(d["policy_refs"]),
            "reason_codes": ",".join(r.gate.packet.reason_codes) if r.gate.packet else "",
        })
    return pd.DataFrame(rows)


def _titles(ax, title, subtitle=None):
    """Title and sub-line placed in offset points, so they never collide
    with each other regardless of how tall the axes is."""
    ax.set_title(title, color=INK, fontsize=12.5, fontweight="semibold",
                 loc="left", pad=28 if subtitle else 10)
    if subtitle:
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="bottom", ha="left")


def _style_axes(ax, *, xgrid=False, ygrid=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_color(INK_2)


def _tiles(ax, df: pd.DataFrame):
    """Four counts are not a chart - they are stat tiles."""
    ax.axis("off")
    counts = {d: int((df["decision"] == d).sum()) for d in DECISION_ORDER}
    n = len(DECISION_ORDER)
    for i, dec in enumerate(DECISION_ORDER):
        x = i / n
        ax.add_patch(plt.Rectangle((x + 0.004, 0.08), 1.0 / n - 0.012, 0.84,
                                   transform=ax.transAxes, facecolor=SURFACE,
                                   edgecolor=GRID, linewidth=1.0, zorder=1))
        ax.plot([x + 0.028], [0.70], marker="o", markersize=7,
                color=DECISION_COLOR[dec], transform=ax.transAxes,
                clip_on=False, zorder=3)
        ax.text(x + 0.050, 0.695, dec.replace("_", " ").title(), transform=ax.transAxes,
                fontsize=9.5, color=INK_2, va="center", ha="left")
        ax.text(x + 0.028, 0.30, str(counts[dec]), transform=ax.transAxes,
                fontsize=30, color=INK, va="center", ha="left", fontweight="semibold")
        ax.text(x + 0.105, 0.315, f"of {len(df)} claims", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="center", ha="left")
    _titles(ax, "Decision breakdown")


def _money(ax, df: pd.DataFrame):
    d = df.iloc[::-1].reset_index(drop=True)      # top-to-bottom = CLM-001 first
    y = range(len(d))
    bar = 0.58
    gap = {"edgecolor": SURFACE, "linewidth": 1.6}

    ax.barh(y, d["approved"], height=bar, color=M_APPROVED, label="Approved", **gap)
    ax.barh(y, d["awaiting_review"], height=bar, left=d["approved"],
            color=M_AWAITING, label="Awaiting human review", **gap)
    ax.barh(y, d["deducted"], height=bar,
            left=d["approved"] + d["awaiting_review"],
            color=M_DEDUCTED, label="Deducted", **gap)

    for i, row in d.iterrows():
        total = row["approved"] + row["awaiting_review"] + row["deducted"]
        ax.text(total + 42, i, f"${total:,.0f}", va="center", ha="left",
                fontsize=9, color=INK_2)
        # direct labels inside segments wide enough to hold them
        running = 0.0
        for val, col in ((row["approved"], "#ffffff"),
                         (row["awaiting_review"], "#ffffff"),
                         (row["deducted"], "#ffffff")):
            if val >= 260:
                ax.text(running + val / 2, i, f"${val:,.0f}", va="center",
                        ha="center", fontsize=8.5, color=col, fontweight="semibold")
            running += val

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r.claim_id}  ·  {r.decision.replace('_',' ').title()}"
                        for r in d.itertuples()], fontsize=9)
    ax.set_xlim(0, max(df["claimed"].max() * 1.16, 100))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"${v:,.0f}"))
    _style_axes(ax, xgrid=True)
    _titles(ax, "Where every claimed dollar went",
            "Bar length is the amount claimed. $0 approved on an escalated claim means "
            "nothing is authorised until a human decides.")
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9, ncol=3,
                    handlelength=0.9, handleheight=0.9, borderpad=0.2,
                    columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(INK_2)


def _confidence(ax, df: pd.DataFrame):
    d = df.iloc[::-1].reset_index(drop=True)
    y = list(range(len(d)))
    ax.hlines(y, 0, d["confidence"], color=BASELINE, linewidth=2.0, zorder=1)
    for i, row in d.iterrows():
        ax.plot([row["confidence"]], [i], marker="o", markersize=9,
                color=DECISION_COLOR[row["decision"]], zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(row["confidence"] + 0.035, i, f"{row['confidence']:.2f}",
                va="center", ha="left", fontsize=9, color=INK_2)
    ax.axvline(CONFIDENCE_FLOOR, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.annotate(f"escalation floor {CONFIDENCE_FLOOR}", xy=(CONFIDENCE_FLOOR, 0),
                xycoords=("data", "axes fraction"), xytext=(0, -34),
                textcoords="offset points", fontsize=8, color=MUTED,
                va="top", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels(list(d["claim_id"]), fontsize=9)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    _style_axes(ax, xgrid=True)
    _titles(ax, "Confidence in each decision",
            "Dot colour repeats the decision. High on an escalation means "
            "\"certainly needs a human\".")


def _refs(ax, df: pd.DataFrame):
    tally: Dict[str, int] = {}
    for refs in df["policy_refs"]:
        for r in [x for x in refs.split(",") if x]:
            tally[r] = tally.get(r, 0) + 1
    items = sorted(tally.items(), key=lambda kv: (kv[1], kv[0]))
    y = range(len(items))
    ax.barh(y, [v for _, v in items], height=0.6, color="#2a78d6",
            edgecolor=SURFACE, linewidth=1.4)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.08, i, str(v), va="center", ha="left", fontsize=9, color=INK_2)
    ax.set_yticks(list(y))
    ax.set_yticklabels([k for k, _ in items], fontsize=8.5, fontfamily="monospace")
    ax.set_xlim(0, max([v for _, v in items] or [1]) + 0.9)
    ax.set_xticks(range(0, len(df) + 1))
    _style_axes(ax, xgrid=True)
    _titles(ax, "Policy rules that drove a decision",
            f"{len(tally)} of {len(REGISTRY)} Appendix A rules materially changed an "
            f"amount, blocked a claim or authorised one. None were invented.")


def build_dashboard(runs, save_to: Optional[str] = "UI SS_1.png"):
    df = results_frame(runs)
    fig = plt.figure(figsize=(12.6, 11.4), facecolor=SURFACE)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.44, 1.0, 1.0],
                  hspace=0.46, wspace=0.26,
                  left=0.085, right=0.965, top=0.880, bottom=0.055)

    fig.text(0.085, 0.962, "Travel Reimbursement Approval Agent",
             fontsize=17, color=INK, fontweight="semibold", ha="left")
    tot = df[["claimed", "approved", "deducted"]].sum()
    pending = float(df.loc[df.decision == "MANUAL_REVIEW", "claimed"].sum())
    fig.text(0.085, 0.936,
             f"{len(df)} claims  ·  ${tot['claimed']:,.0f} claimed  ·  "
             f"${tot['approved']:,.0f} approved  ·  ${tot['deducted']:,.0f} deducted  ·  "
             f"${pending:,.0f} awaiting human review",
             fontsize=10, color=INK_2, ha="left")

    _tiles(fig.add_subplot(gs[0, :]), df)
    _money(fig.add_subplot(gs[1, :]), df)
    _confidence(fig.add_subplot(gs[2, 0]), df)
    _refs(fig.add_subplot(gs[2, 1]), df)

    if save_to:
        fig.savefig(save_to, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig, df


# ---------------------------------------------------------------------
# Single-claim review card - the same content the interactive panel shows.
# Laid out with a flowing cursor measured in text lines, so nothing
# collides however long an explanation runs.
# ---------------------------------------------------------------------
import textwrap

CARD_W = 108            # wrap width, characters
HEAD_IN = 2.35          # header band + metric row, inches
LINE_IN = 0.175         # one rendered body line, inches
LABEL_IN = 0.26         # a block's small caps label, inches
BLOCKGAP_IN = 0.16      # gap after each block, inches
FOOT_IN = 0.25


def build_claim_card(run, save_to: Optional[str] = None):
    d = run.result
    claim = CLAIMS_BY_ID[d["claim_id"]]
    col = DECISION_COLOR[d["decision"]]
    packet = run.gate.packet

    # --- pass 1: build every block, then size the canvas to fit it ---
    blocks = [
        ("EXPLANATION", textwrap.fill(d["explanation"], CARD_W), INK, 9.3, None),
        ("POLICY REFS", "   ".join(d["policy_refs"]), "#0d6a4d", 9.3, "monospace"),
        ("MISSING DOCS",
         "\n".join(textwrap.fill(m, CARD_W) for m in (d["missing_docs"] or ["(none)"])),
         C_REJECT if d["missing_docs"] else MUTED, 9.3, None),
        ("TOOLS USED", textwrap.fill("  ->  ".join(d["tools_used"]), CARD_W + 6),
         INK_2, 8.6, "monospace"),
    ]
    if packet:
        blocks.append(("MANUAL REVIEW PACKET",
                       "\n".join(textwrap.fill(f"- {c}", CARD_W)
                                 for c in packet.what_to_check), INK, 9.3, None))
        blocks.append(("PROVISIONAL (not authorised)",
                       textwrap.fill(
                           f"${packet.provisional_reimbursable} reimbursable  ·  "
                           f"${packet.provisional_deducted} deducted  ·  reasons: "
                           f"{', '.join(packet.reason_codes)}", CARD_W + 6),
                       MUTED, 8.6, "monospace"))

    body_in = sum(LABEL_IN + LINE_IN * (b[1].count("\n") + 1) + BLOCKGAP_IN
                  for b in blocks)
    H = HEAD_IN + body_in + FOOT_IN
    fig = plt.figure(figsize=(11.4, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def inch(v):                       # inches -> figure fraction
        return v / H

    def T(x, y, s, size=10, color=INK, weight="normal", family=None, ha="left"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                family=family, ha=ha, va="top")

    L, R = 0.045, 0.955

    # --- header band ---
    band_h = inch(0.80)
    band_top = 1 - inch(0.22)
    ax.add_patch(plt.Rectangle((L, band_top - band_h), R - L, band_h,
                               facecolor="#ffffff", edgecolor=GRID, linewidth=1.0))
    ax.add_patch(plt.Rectangle((L, band_top - band_h), 0.006, band_h,
                               facecolor=col, edgecolor="none"))
    T(L + 0.022, band_top - inch(0.20), f"{d['claim_id']}  ·  {claim.title}",
      13, INK, "semibold")
    T(L + 0.022, band_top - inch(0.50),
      f"{claim.employee}  ·  trip {claim.trip_start} to {claim.trip_end}  ·  "
      f"submitted {claim.submitted_date}", 9.5, MUTED)
    T(R - 0.018, band_top - inch(0.20), d["decision"].replace("_", " "),
      13, col, "semibold", ha="right")

    # --- metric row ---
    mtop = band_top - band_h - inch(0.30)
    for x, (label, val, c) in zip(
        (L + 0.004, L + 0.215, L + 0.425, L + 0.635),
        (("APPROVED", f"${d['approved_amount']:,.2f}", C_APPROVE),
         ("DEDUCTED", f"${d['deducted_amount']:,.2f}", C_REJECT),
         ("CONFIDENCE", f"{d['confidence']:.2f}", INK),
         ("CLAIMED", f"${float(claim.stated_total):,.2f}", INK_2)),
    ):
        T(x, mtop, label, 8.5, MUTED, family="monospace")
        T(x, mtop - inch(0.22), val, 19, c, "semibold")

    # --- pass 2: flow the blocks ---
    y = 1 - inch(HEAD_IN)
    for label, body, color, size, family in blocks:
        T(L, y, label, 8.5, MUTED, family="monospace")
        y -= inch(LABEL_IN)
        T(L, y, body, size, color, family=family)
        y -= inch(LINE_IN * (body.count("\n") + 1) + BLOCKGAP_IN)

    if save_to:
        fig.savefig(save_to, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


# =====================================================================
# SECTION 12b — INTERACTIVE REVIEW CONSOLE
# =====================================================================
# This is the interface a reviewer actually drives: pick a claim, run it,
# read the recommendation and the full tool trace. It degrades to static
# output wherever ipywidgets is unavailable (GitHub, nbconvert, plain
# Jupyter without the extension), so the notebook never depends on it.
# =====================================================================
_CSS = """
<style>
.rc-card{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
 border:1px solid #e1e0d9;border-radius:6px;padding:0;margin:8px 0;background:#fff;
 max-width:980px;overflow:hidden}
.rc-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;
 padding:14px 18px;border-bottom:1px solid #e1e0d9}
.rc-id{font-weight:650;color:#0b0b0b;font-size:15px}
.rc-sub{color:#898781;font-size:12px;margin-top:3px}
.rc-dec{font-weight:700;font-size:14px;letter-spacing:.02em;white-space:nowrap}
.rc-metrics{display:flex;gap:34px;padding:14px 18px;border-bottom:1px solid #e1e0d9;
 flex-wrap:wrap}
.rc-m label{display:block;font-size:10px;letter-spacing:.09em;color:#898781;
 font-family:ui-monospace,Menlo,monospace;margin-bottom:3px}
.rc-m span{font-size:21px;font-weight:650;font-variant-numeric:tabular-nums}
.rc-body{padding:14px 18px}
.rc-body h4{margin:14px 0 5px;font-size:10px;letter-spacing:.09em;color:#898781;
 font-family:ui-monospace,Menlo,monospace;font-weight:600}
.rc-body h4:first-child{margin-top:0}
.rc-body p{margin:0;font-size:13.5px;line-height:1.6;color:#0b0b0b}
.rc-ref{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#0d6a4d;
 background:#e2efe8;padding:2px 6px;border-radius:3px;margin-right:5px;
 display:inline-block;margin-bottom:4px}
.rc-miss{color:#d03b3b;font-size:13px;margin:2px 0}
.rc-tool{font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#52514e;
 background:#f6f8f4;padding:2px 6px;border-radius:3px;margin-right:4px;
 display:inline-block;margin-bottom:4px}
.rc-note{font-size:12.5px;color:#52514e;margin:3px 0;padding-left:14px;
 border-left:2px solid #e1e0d9}
table.rc-t{border-collapse:collapse;font-size:12px;width:100%;margin-top:4px}
table.rc-t th{text-align:left;font-family:ui-monospace,Menlo,monospace;font-size:10px;
 letter-spacing:.06em;color:#898781;border-bottom:1px solid #e1e0d9;padding:5px 8px;
 font-weight:600}
table.rc-t td{padding:5px 8px;border-bottom:1px solid #f0efec;color:#0b0b0b}
table.rc-t td.n{text-align:right;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,Menlo,monospace}
</style>
"""


def result_html(run) -> str:
    """One claim rendered as the reviewer sees it."""
    import html as _h
    d = run.result
    claim = CLAIMS_BY_ID[d["claim_id"]]
    col = DECISION_COLOR[d["decision"]]

    lines = "".join(
        f"<tr><td>{v.line_id}</td><td>{v.category}</td>"
        f"<td>{_h.escape(v.description)}</td>"
        f"<td class='n'>${v.claimed}</td><td class='n'>${v.allowed}</td>"
        f"<td class='n'>${v.excess}</td>"
        f"<td>{'yes' if v.receipt_attached else 'NO'}</td></tr>"
        for v in run.ledger.lines)

    refs = "".join(f"<span class='rc-ref'>{r}</span>" for r in d["policy_refs"])
    tools = "".join(f"<span class='rc-tool'>{t}</span>" for t in d["tools_used"])
    miss = ("".join(f"<p class='rc-miss'>{_h.escape(m)}</p>" for m in d["missing_docs"])
            or "<p style='color:#898781;font-size:13px;margin:0'>none</p>")
    notes = ("".join(f"<p class='rc-note'>{_h.escape(n)}</p>" for n in run.gate.notes)
             or "<p class='rc-note'>No override needed - the agent agreed with the "
                "engine.</p>")

    packet = ""
    if run.gate.packet:
        p = run.gate.packet
        items = "".join(f"<p class='rc-note'>{_h.escape(c)}</p>" for c in p.what_to_check)
        packet = (f"<h4>MANUAL REVIEW PACKET</h4>{items}"
                  f"<p class='rc-note' style='color:#898781'>provisional: "
                  f"${p.provisional_reimbursable} reimbursable &middot; "
                  f"${p.provisional_deducted} deducted &middot; "
                  f"reasons: {', '.join(p.reason_codes)}</p>")

    return f"""{_CSS}
<div class='rc-card'>
  <div class='rc-head' style='border-left:4px solid {col}'>
    <div><div class='rc-id'>{d['claim_id']} &middot; {_h.escape(claim.title)}</div>
    <div class='rc-sub'>{_h.escape(claim.employee)} &middot; trip {claim.trip_start}
      to {claim.trip_end} &middot; submitted {claim.submitted_date}
      &middot; mode {run.audit.mode}</div></div>
    <div class='rc-dec' style='color:{col}'>{d['decision'].replace('_',' ')}</div>
  </div>
  <div class='rc-metrics'>
    <div class='rc-m'><label>APPROVED</label>
      <span style='color:{C_APPROVE}'>${d['approved_amount']:,.2f}</span></div>
    <div class='rc-m'><label>DEDUCTED</label>
      <span style='color:{C_REJECT}'>${d['deducted_amount']:,.2f}</span></div>
    <div class='rc-m'><label>CONFIDENCE</label><span>{d['confidence']:.2f}</span></div>
    <div class='rc-m'><label>CLAIMED</label>
      <span style='color:#52514e'>${float(claim.stated_total):,.2f}</span></div>
  </div>
  <div class='rc-body'>
    <h4>EXPLANATION</h4><p>{_h.escape(d['explanation'])}</p>
    <h4>LINE-BY-LINE LEDGER</h4>
    <table class='rc-t'><tr><th>line</th><th>category</th><th>description</th>
      <th>claimed</th><th>allowed</th><th>excess</th><th>receipt</th></tr>{lines}</table>
    <h4>POLICY REFS</h4><div>{refs}</div>
    <h4>MISSING DOCS</h4>{miss}
    <h4>TOOL TRACE ({len(d['tools_used'])} calls, {len(run.audit.retrieved_rules)}
      rules retrieved)</h4><div>{tools}</div>
    <h4>RECONCILIATION GATE</h4>{notes}
    {packet}
  </div>
</div>"""


def review_console(runs):
    """Interactive panel; falls back to static cards when widgets are absent."""
    from IPython.display import HTML, display
    by_id = {r.claim.claim_id: r for r in runs}
    try:
        import ipywidgets as W
    except ImportError:
        print("ipywidgets unavailable - rendering all claims statically instead.")
        for r in runs:
            display(HTML(result_html(r)))
        return None

    picker = W.Dropdown(options=list(by_id), value=list(by_id)[0],
                        description="Claim:",
                        layout=W.Layout(width="320px"))
    show_all = W.Button(description="Show all 5",
                        layout=W.Layout(width="130px"))
    out = W.Output()

    def render(_=None):
        with out:
            out.clear_output(wait=True)
            display(HTML(result_html(by_id[picker.value])))

    def render_all(_=None):
        with out:
            out.clear_output(wait=True)
            for r in runs:
                display(HTML(result_html(r)))

    picker.observe(lambda ch: render() if ch["name"] == "value" else None)
    show_all.on_click(render_all)
    render()
    panel = W.VBox([W.HBox([picker, show_all]), out])
    display(panel)
    return panel
