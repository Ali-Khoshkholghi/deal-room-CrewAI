"""Milestone 5: wraps the M3/M4 hierarchical crew in a CrewAI `Flow`,
adding a conditional second ("deep dive") pass when the first pass's own
`InvestmentMemo.confidence` is low or its `recommendation` is "needs more
diligence" — the two signals the crew already produces to flag its own
uncertainty, rather than a separate judgment this Flow has to make itself.

SCAFFOLDED, UNTESTED as of 2026-08-06: construct-checked only (the `Flow`
instantiates, its state schema validates, `visualize_flow_structure()`
renders the start -> router -> branch topology) — never run against a
live `kickoff()`. Cerebras quota was exhausted (see README's Milestone 4
"Memory cost fix" section) when this was written, so real verification —
does the router actually pick the right branch on a real low-confidence
memo, does the deep-dive crew produce a materially different memo, does
the state end up correctly populated either way — is deferred to the same
quota-reset re-run as Milestone 4's memory-cost levers. See README's
"Milestone 5: CrewAI Flows (scaffolded, untested)" section.
"""

from __future__ import annotations

from crewai.flow.flow import Flow, FlowState, listen, router, start

from deal_room.crew import DealRoomCrew, kickoff_with_retry
from deal_room.models import InvestmentMemo


class DealRoomFlowState(FlowState):
    """State tracked across `DealRoomFlow`'s steps.

    Subclasses CrewAI's `FlowState` rather than a plain `pydantic.BaseModel`
    — required, not stylistic: `Flow` validates on construction that its
    state model has an `id` field (`FlowState` provides one, a UUID
    generated per run), and rejects a plain `BaseModel` state with a real
    `pydantic.ValidationError` at instantiation time. Caught by
    construct-checking this class before ever attempting a live run — see
    README's Milestone 5 section.

    `final_memo` starts as a copy of `initial_memo` (set at the end of
    `run_initial_analysis`) and is only overwritten if the deep-dive branch
    actually runs — so a caller reading `final_memo` after `kickoff()`
    always gets a real memo regardless of which branch executed, without
    having to check `deep_dive_triggered` first to know which field is
    valid.
    """

    company_info: str = ""
    # Accepted per Milestone 5's spec, but not yet wired to anything: the
    # underlying M3/M4 crew's `PitchDeckReaderTool` is constructed with a
    # hardcoded `default_file_path=str(SAMPLE_DECK_PATH)` per specialist
    # agent (see `crew.py`), not a path threaded through `Task` inputs like
    # `company_info` is. Making this field actually override which deck
    # gets read is a real change to `crew.py`'s tool construction, out of
    # scope for this scaffolding pass.
    pitch_deck_path: str | None = None
    initial_memo: InvestmentMemo | None = None
    deep_dive_triggered: bool = False
    final_memo: InvestmentMemo | None = None


class DealRoomFlow(Flow[DealRoomFlowState]):
    """Milestone 5: single conditional branch on top of the M3/M4 crew.

    Topology: `run_initial_analysis` (`@start`) -> `decide_deep_dive`
    (`@router`, reads `state.initial_memo`) -> either `run_deep_dive` or
    `finalize_without_deep_dive` (`@listen`, mutually exclusive on the
    router's return value).

    Deliberately minimal for this milestone: the deep-dive branch is a
    second full crew pass with an appended task instruction, not a
    differently-shaped crew (e.g. more specialist re-queries, a longer
    research budget, extra tools). Real deep-dive behavior — whether one
    extra instruction sentence actually changes what the manager delegates
    or just produces a superficially-reworded memo — is exactly the kind
    of claim this project's own M3/M4 sections warn against assuming
    without live evidence (see README's "Re-query behavior: negative
    result"). Treat this class's second pass as a structural stub, not a
    verified behavior, until it's actually been run.
    """

    initial_state: type[DealRoomFlowState] = DealRoomFlowState

    @start()
    def run_initial_analysis(self) -> None:
        crew = DealRoomCrew()
        result = kickoff_with_retry(
            crew.crew(), {"company_info": self.state.company_info}
        )
        memo = result.pydantic
        self.state.initial_memo = memo
        self.state.final_memo = memo  # default; overwritten if deep dive runs

    @router(run_initial_analysis)
    def decide_deep_dive(self) -> str:
        """Route based on the first pass's own uncertainty signals.

        No `initial_memo` (e.g. `output_pydantic` failed to parse — a real,
        observed failure mode for this crew, see README's M3 findings) is
        treated as its own reason to skip the deep dive rather than crash
        the flow: there's nothing to re-diligence against.
        """
        memo = self.state.initial_memo
        if memo is None:
            return "skip_deep_dive"
        if memo.confidence == "low" or memo.recommendation == "needs more diligence":
            return "deep_dive"
        return "skip_deep_dive"

    @listen("deep_dive")
    def run_deep_dive(self) -> str | None:
        """Stub deep-dive pass: a second full crew kickoff with the prior
        assessment's uncertainty appended to the same `company_info` input.

        Full implementation (e.g. actually feeding the prior memo's
        specific red flags back in as targeted follow-up questions, rather
        than one generic sentence) is explicitly deferred — this exists to
        make the branching logic and state schema real and testable now,
        not to be a finished second-pass design.
        """
        self.state.deep_dive_triggered = True
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
        crew = DealRoomCrew()
        result = kickoff_with_retry(crew.crew(), {"company_info": deep_dive_info})
        self.state.final_memo = result.pydantic
        return self.state.final_memo

    @listen("skip_deep_dive")
    def finalize_without_deep_dive(self) -> str | None:
        # state.final_memo already set to state.initial_memo in
        # run_initial_analysis -- nothing to do but return it, so
        # kickoff() yields the same shape (an InvestmentMemo | None)
        # regardless of which branch actually ran.
        return self.state.final_memo
