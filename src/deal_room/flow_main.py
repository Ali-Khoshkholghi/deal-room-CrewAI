"""Milestone 5 entry point — mirrors `main.py`'s structure and reuses the
same test pitch, but drives `DealRoomFlow` instead of `DealRoomCrew`
directly.

Live-tested 2026-08-07 (router mechanism confirmed correct on two real
crew passes; see README's Milestone 5 section for the full writeup,
including why the "skip the deep dive" branch remains unconfirmed against
a real non-triggering memo). Updated 2026-08-08 for two fixes: `deck_path`
is now threaded through as an optional argument to `run()` (previously
`DealRoomFlowState.pitch_deck_path` was accepted but unused), and the
result printout below now surfaces `pass_count` and `needs_human_review`
so the escalation-cap fix (`flow.py`'s `MAX_PASSES`) is visible to a
caller, not just present in state.
"""

from dotenv import load_dotenv

from deal_room.flow import DealRoomFlow

load_dotenv()

# Same test pitch as Milestones 1-4, for continuity.
COMPANY_INFO = """
Company: Brightledger
B2B SaaS platform for automated invoice reconciliation, targeting mid-market 
finance teams (50-500 employees). Seed stage, 14 months post-launch. 
MRR $42,000 (up from $9,000 eight months ago). Churn 4% monthly. 
Team of 6, founders previously at Stripe and Brex. Raised $2.5M at $12M 
pre-money. 38 paying customers, average contract $1,100/mo.
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


def run(deck_path: str | None = None):
    flow = DealRoomFlow()
    flow.kickoff(inputs={"company_info": COMPANY_INFO, "pitch_deck_path": deck_path})

    print("\n" + "=" * 70)
    print("MILESTONE 5 FLOW RESULT")
    print("=" * 70)
    print(f"Pass count          : {flow.state.pass_count}")
    print(f"Deep dive triggered : {flow.state.deep_dive_triggered}")
    print(f"Needs human review  : {flow.state.needs_human_review}")
    if flow.state.needs_human_review:
        print(
            "\n*** STILL UNCERTAIN AFTER ESCALATION *** -- the deep dive "
            "ran (the one extra pass MAX_PASSES allows) and its own result "
            "was still confidence='low' or recommendation='needs more "
            "diligence'. This flow does not auto-escalate further; treat "
            "final_memo below as provisional, pending human review, not a "
            "resolved recommendation."
        )

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
