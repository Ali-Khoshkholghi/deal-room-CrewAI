"""Milestone 5 entry point — mirrors `main.py`'s structure and reuses the
same test pitch, but drives `DealRoomFlow` instead of `DealRoomCrew`
directly.

NOT RUN as of 2026-08-06 — Cerebras quota was exhausted (see README's
Milestone 4 "Memory cost fix" section) when this was written. Live
verification (does the router actually trigger the deep dive on a real
low-confidence memo, does `state.final_memo` end up correctly populated
either way) is deferred to the same quota-reset re-run as Milestone 4's
memory-cost levers — see README's "Milestone 5: CrewAI Flows (scaffolded,
untested)" section. `deal_room.flow`'s own module docstring and
`DealRoomFlow`'s class docstring have been construct-checked (the Flow
instantiates, its state schema validates, its topology renders via
`build_flow_structure()`/`plot()`) but never kicked off live.
"""

from dotenv import load_dotenv

from deal_room.flow import DealRoomFlow

load_dotenv()

# Same test pitch as Milestones 1-4, for continuity.
COMPANY_INFO = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""


def _print_memo(label: str, memo) -> None:
    if memo is None:
        print(f"\n{label}: None (no parseable InvestmentMemo)")
        return
    print("\n" + "=" * 70)
    print(f"{label}")
    print("=" * 70)
    print(f"Company summary   : {memo.company_summary}")
    print(f"Financial summary : {memo.financial_summary}")
    print(f"Market summary    : {memo.market_summary}")
    print(f"Technical summary : {memo.technical_summary}")
    print(f"Risk summary      : {memo.risk_summary}")
    print(f"Key red flags     : {memo.key_red_flags}")
    print(f"Recommendation    : {memo.recommendation}")
    print(f"Confidence        : {memo.confidence}")


def run():
    flow = DealRoomFlow()
    flow.kickoff(inputs={"company_info": COMPANY_INFO})

    print("\n" + "=" * 70)
    print("MILESTONE 5 FLOW RESULT")
    print("=" * 70)
    print(f"Deep dive triggered: {flow.state.deep_dive_triggered}")

    _print_memo("Initial memo (first pass)", flow.state.initial_memo)
    if flow.state.deep_dive_triggered:
        _print_memo("Final memo (after deep dive)", flow.state.final_memo)
    else:
        print(
            "\nNo deep dive: final memo is the same object as the initial "
            "memo (confidence was not 'low' and recommendation was not "
            "'needs more diligence')."
        )


if __name__ == "__main__":
    run()
