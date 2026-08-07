"""Milestone 5: wraps the M3/M4 hierarchical crew in a CrewAI `Flow`,
adding a conditional second ("deep dive") pass when the first pass's own
`InvestmentMemo.confidence` is low or its `recommendation` is "needs more
diligence" — the two signals the crew already produces to flag its own
uncertainty, rather than a separate judgment this Flow has to make itself.

Live-tested 2026-08-07: the router mechanism itself is confirmed correct
against two real crew passes (reads the real memo, applies its documented
condition, sets state accordingly). The "skip the deep dive" branch is
still unconfirmed against a real non-triggering memo — both a strong and
weak synthetic test pitch triggered it, partly because specialists
reflexively re-invoke `PitchDeckReaderTool` regardless of input
completeness, pulling in the fixed sample deck's content. See README's
Milestone 5 section for the full writeup, including a real CrewAI `Flow`
subclassing gotcha discovered along the way.

Updated 2026-08-08, two fixes:

1. `pitch_deck_path` is now actually wired through to `DealRoomCrew`
   (previously accepted in the state schema but never used) — combined
   with `crew.py`'s fix making `PitchDeckReaderTool` attachment
   conditional on a deck being provided at all, rather than always
   attached and reflexively invoked regardless of relevance.
2. A hard cap on deep-dive escalation: at most 2 total crew passes
   (initial + one deep dive), tracked via `state.pass_count` and enforced
   defensively inside `decide_deep_dive` itself (not just by the absence
   of a loop in the topology — see that method's docstring). If the
   deep-dive pass's own result is STILL uncertain
   (`confidence == "low"` or `recommendation == "needs more diligence"`),
   the flow does not attempt further escalation; it sets
   `state.needs_human_review = True` and returns the deep-dive result as
   final, explicitly marked as still-uncertain rather than silently
   presented as a normal completed result.
"""

from __future__ import annotations

from crewai.flow.flow import Flow, FlowState, listen, router, start

from deal_room.crew import DealRoomCrew, kickoff_with_retry
from deal_room.models import InvestmentMemo

# Total crew passes allowed in one flow run: the initial pass, plus at most
# one deep dive. Not "no more than N deep dives" -- there is only ever at
# most one deep dive by topology (see decide_deep_dive's docstring), so
# MAX_PASSES=2 is the entire cap, not a loop counter.
MAX_PASSES = 2


class DealRoomFlowState(FlowState):
    """State tracked across `DealRoomFlow`'s steps.

    Subclasses CrewAI's `FlowState` rather than a plain `pydantic.BaseModel`
    — required, not stylistic: `Flow` validates on construction that its
    state model has an `id` field (`FlowState` provides one, a UUID
    generated per run), and rejects a plain `BaseModel` state with a real
    `pydantic.ValidationError` at instantiation time.

    `final_memo` starts as a copy of `initial_memo` (set at the end of
    `run_initial_analysis`) and is only overwritten if the deep-dive branch
    actually runs — so a caller reading `final_memo` after `kickoff()`
    always gets a real memo regardless of which branch executed, without
    having to check `deep_dive_triggered` first to know which field is
    valid. `needs_human_review` is the caller's signal that `final_memo`,
    even after the deep dive, is still not confident/decided — it should
    be checked alongside `final_memo`, not assumed false just because a
    memo came back.
    """

    company_info: str = ""
    # Now actually wired through to DealRoomCrew(deck_path=...) in
    # run_initial_analysis/run_deep_dive below (as of 2026-08-08) --
    # previously accepted here but never used; the underlying crew's
    # PitchDeckReaderTool attachment is conditional on this being set (see
    # crew.py's __init__ and _deck_tools()), not always-on.
    pitch_deck_path: str | None = None
    initial_memo: InvestmentMemo | None = None
    deep_dive_triggered: bool = False
    final_memo: InvestmentMemo | None = None
    # How many crew passes have actually run (1 after the initial pass, 2
    # after a deep dive). Read by decide_deep_dive to enforce MAX_PASSES.
    pass_count: int = 0
    # True iff a deep dive ran AND its own result was still uncertain
    # (confidence == "low" or recommendation == "needs more diligence").
    # The flow does not auto-escalate further when this is set -- see
    # run_deep_dive's docstring.
    needs_human_review: bool = False


class DealRoomFlow(Flow[DealRoomFlowState]):
    """Milestone 5: single conditional branch on top of the M3/M4 crew,
    now with a hard cap on escalation (2026-08-08).

    Topology: `run_initial_analysis` (`@start`) -> `decide_deep_dive`
    (`@router`, reads `state.initial_memo` and `state.pass_count`) ->
    either `run_deep_dive` or `finalize_without_deep_dive` (`@listen`,
    mutually exclusive on the router's return value). There is no path
    from `run_deep_dive` back to `decide_deep_dive` -- the topology itself
    only ever allows one deep dive, and `decide_deep_dive`'s own
    `pass_count` check (below) makes that a defensive, self-enforcing
    property of the state machine rather than just an absence of wiring.

    Deliberately minimal for this milestone: the deep-dive branch is a
    second full crew pass with an appended task instruction, not a
    differently-shaped crew (e.g. more specialist re-queries, a longer
    research budget, extra tools). Real deep-dive behavior — whether one
    extra instruction sentence actually changes what the manager delegates
    or just produces a superficially-reworded memo — is exactly the kind
    of claim this project's own M3/M4 sections warn against assuming
    without live evidence (see README's "Re-query behavior: negative
    result"). Treat this class's second pass as a structural stub, not a
    verified behavior.
    """

    initial_state: type[DealRoomFlowState] = DealRoomFlowState

    @start()
    def run_initial_analysis(self) -> None:
        crew = DealRoomCrew(deck_path=self.state.pitch_deck_path)
        result = kickoff_with_retry(
            crew.crew(), {"company_info": self.state.company_info}
        )
        memo = result.pydantic
        self.state.initial_memo = memo
        self.state.final_memo = memo  # default; overwritten if deep dive runs
        self.state.pass_count = 1

    @router(run_initial_analysis)
    def decide_deep_dive(self) -> str:
        """Route based on the first pass's own uncertainty signals, capped
        by `state.pass_count`.

        No `initial_memo` (e.g. `output_pydantic` failed to parse — a real,
        observed M3 failure mode) is treated as its own reason to skip the
        deep dive rather than crash the flow: there's nothing to
        re-diligence against.

        The `pass_count >= MAX_PASSES` check is defensive, not load-bearing
        under the current topology (there's no edge from `run_deep_dive`
        back to this router, so it can only ever fire once per kickoff()
        as currently wired) -- but it makes the "at most one deep dive"
        rule an explicit, enforced property of the state machine, not just
        an accident of the current graph shape. If a future change ever
        did wire a loop back here, this line is what actually stops it.
        """
        if self.state.pass_count >= MAX_PASSES:
            return "skip_deep_dive"
        memo = self.state.initial_memo
        if memo is None:
            return "skip_deep_dive"
        if memo.confidence == "low" or memo.recommendation == "needs more diligence":
            return "deep_dive"
        return "skip_deep_dive"

    @listen("deep_dive")
    def run_deep_dive(self) -> InvestmentMemo | None:
        """Stub deep-dive pass: a second full crew kickoff with the prior
        assessment's uncertainty appended to the same `company_info` input.

        Full implementation (e.g. actually feeding the prior memo's
        specific red flags back in as targeted follow-up questions, rather
        than one generic sentence) is explicitly deferred — this exists to
        make the branching logic and state schema real and testable now,
        not to be a finished second-pass design.

        Escalation cap (2026-08-08): this is the LAST pass allowed by
        MAX_PASSES. If this pass's own result is still uncertain by the
        same criteria that triggered it in the first place, this method
        does not try to escalate again -- it sets
        `state.needs_human_review = True` and returns the deep-dive memo
        as `final_memo` regardless, explicitly flagged rather than handed
        back looking like a normal, resolved result.
        """
        prior = self.state.initial_memo
        prior_confidence = prior.confidence if prior else "unknown"
        prior_recommendation = prior.recommendation if prior else "unknown"
        deep_dive_info = (
            f"{self.state.company_info}\n\n"
            "Additional instruction: a prior assessment of this company "
            f"came back with confidence '{prior_confidence}' and "
            f"recommendation '{prior_recommendation}'. Conduct additional "
            "research given that prior uncertainty before finalizing this "
            "memo."
        )
        crew = DealRoomCrew(deck_path=self.state.pitch_deck_path)
        result = kickoff_with_retry(crew.crew(), {"company_info": deep_dive_info})
        deep_dive_memo = result.pydantic

        self.state.deep_dive_triggered = True
        self.state.pass_count = MAX_PASSES
        self.state.final_memo = deep_dive_memo

        still_uncertain = deep_dive_memo is not None and (
            deep_dive_memo.confidence == "low"
            or deep_dive_memo.recommendation == "needs more diligence"
        )
        if still_uncertain:
            self.state.needs_human_review = True

        return self.state.final_memo

    @listen("skip_deep_dive")
    def finalize_without_deep_dive(self) -> InvestmentMemo | None:
        # state.final_memo already set to state.initial_memo in
        # run_initial_analysis -- nothing to do but return it, so
        # kickoff() yields the same shape (an InvestmentMemo | None)
        # regardless of which branch actually ran.
        return self.state.final_memo
