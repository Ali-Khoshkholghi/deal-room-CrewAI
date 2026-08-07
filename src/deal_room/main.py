import time

from dotenv import load_dotenv

from deal_room.crew import RATE_LIMIT_COOLDOWN_SECONDS, DealRoomCrew, kickoff_with_retry
from deal_room.custom_memory import recall_memo, save_memo

load_dotenv()

# Companies are keyed by this name in custom_memory.py's LanceDB table.
COMPANY_NAME = "Brightledger"

# Same test pitch as Milestones 1-3, for continuity across milestones.
COMPANY_INFO = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""

# A follow-up question about the SAME company (referenced by name only),
# deliberately NOT re-supplying any of the figures above in the QUESTION
# text itself. As of 2026-08-07 this project no longer relies on CrewAI's
# automatic per-agent-step memory to answer it -- `run()` below explicitly
# recalls the prior memo via `custom_memory.recall_memo()` and injects it as
# plain-text context ahead of this query, so the crew has the real figures
# available to read, not to guess or magically retrieve.
FOLLOW_UP_QUERY = """
Following up on Brightledger, the company we discussed earlier -- do not
ask for its founding details, funding amount, team size, or MRR again,
you already have those from before. Just answer: given what you already
know about Brightledger, is the risk profile still "medium", or has
anything changed? Also state what you recall about its MRR and monthly
churn figures from before, if anything.
"""

# Substrings from COMPANY_INFO used to check that kickoff #2's own output
# actually carries forward the figures from the injected prior context,
# rather than ignoring it and hedging anyway.
_RECALL_MARKERS = ["42,000", "9,000", "4%", "1.8M", "11 people", "pre-seed"]
_HEDGE_PHRASES = [
    "insufficient data",
    "not enough information",
    "please provide",
    "need more information",
    "don't have",
    "do not have",
    "no information",
    "unable to determine",
    "cannot determine",
]


def _print_memo(label: str, memo) -> None:
    print("\n" + "=" * 70)
    print(f"FINAL INVESTMENT MEMO (structured) — {label}")
    print("=" * 70)
    print(f"Company summary   : {memo.company_summary}")
    print(f"Financial summary : {memo.financial_summary}")
    print(f"Market summary    : {memo.market_summary}")
    print(f"Technical summary : {memo.technical_summary}")
    print(f"Risk summary      : {memo.risk_summary}")
    print(f"Key red flags     : {memo.key_red_flags}")
    print(f"Recommendation    : {memo.recommendation}")
    print(f"Confidence        : {memo.confidence}")


def _print_stats(label: str, elapsed: float, usage) -> None:
    print("\n" + "=" * 70)
    print(f"EXECUTION STATS — {label}")
    print("=" * 70)
    print(f"Wall-clock time      : {elapsed:.1f}s")
    print(f"Successful requests  : {usage.successful_requests}")
    print(f"Prompt tokens        : {usage.prompt_tokens}")
    print(f"Completion tokens    : {usage.completion_tokens}")
    print(f"Total tokens         : {usage.total_tokens}")
    print(
        "Memory cost          : $0 / 0 requests -- CrewAI's automatic "
        "per-agent-step memory is disabled project-wide (see crew.py); "
        "recall/save below are single explicit calls with zero LLM usage "
        "(see custom_memory.py)."
    )
    print(
        "Cost estimate        : not computed — Cerebras pricing isn't in "
        "CrewAI's built-in cost table for this model; check current "
        "per-token rates at https://cloud.cerebras.ai/pricing and multiply "
        "against the token counts above for an actual figure."
    )


def _memo_text_blob(memo) -> str:
    return " ".join(
        [
            memo.company_summary,
            memo.financial_summary,
            memo.market_summary,
            memo.technical_summary,
            memo.risk_summary,
            memo.recommendation,
            memo.confidence,
            " ".join(memo.key_red_flags),
        ]
    ).lower()


def _prior_context_text(recalled: dict) -> str:
    """Render a `custom_memory.recall_memo()` result as plain text to prepend
    to a new `company_info` input. No framework-level memory involved here --
    this is just an explicit string the crew reads like any other input,
    the same way COMPANY_INFO itself is read."""
    return (
        f"Prior assessment on file for {recalled['company_name']}:\n"
        f"- Company: {recalled['company_summary']}\n"
        f"- Financials: {recalled['financial_summary']}\n"
        f"- Market: {recalled['market_summary']}\n"
        f"- Technical: {recalled['technical_summary']}\n"
        f"- Risk: {recalled['risk_summary']}\n"
        f"- Recommendation: {recalled['recommendation']} "
        f"(confidence: {recalled['confidence']})\n"
    )


def run():
    # --- Kickoff #1: full company info. Recall first (single call, zero
    # LLM cost) in case a prior record already exists from an earlier run
    # of this script -- if so, inject it as plain prior context so the crew
    # doesn't start from zero even on "kickoff #1" of this particular
    # process. ---
    prior1 = recall_memo(COMPANY_NAME)
    company_info_1 = COMPANY_INFO
    if prior1 is not None:
        company_info_1 = _prior_context_text(prior1) + "\n" + COMPANY_INFO
        print(f"Recalled an existing prior record for {COMPANY_NAME!r} -- injecting as context.")

    crew1 = DealRoomCrew()
    crew_obj_1 = crew1.crew()
    start = time.monotonic()
    result1 = kickoff_with_retry(crew_obj_1, {"company_info": company_info_1})
    elapsed1 = time.monotonic() - start
    memo1 = result1.pydantic

    _print_memo("Kickoff #1 (Brightledger, full info)", memo1)
    _print_stats("Kickoff #1", elapsed1, result1.token_usage)

    if memo1 is not None:
        save_memo(COMPANY_NAME, memo1)
        print(f"\nSaved kickoff #1's memo under {COMPANY_NAME!r} (single call, zero LLM cost).")

    print(f"\nCooling down {RATE_LIMIT_COOLDOWN_SECONDS}s before kickoff #2...")
    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)

    # --- Kickoff #2: recall the just-saved memo (single call, zero LLM
    # cost), inject it as plain-text prior context, then ask the follow-up
    # question -- which itself still withholds the figures -- to see
    # whether the crew actually reads and uses the injected context block,
    # rather than ignoring it and hedging anyway. ---
    prior2 = recall_memo(COMPANY_NAME)
    assert prior2 is not None, "expected a prior record to exist after kickoff #1's save_memo()"
    company_info_2 = _prior_context_text(prior2) + "\n" + FOLLOW_UP_QUERY

    crew2 = DealRoomCrew()
    crew_obj_2 = crew2.crew()
    start = time.monotonic()
    result2 = kickoff_with_retry(crew_obj_2, {"company_info": company_info_2})
    elapsed2 = time.monotonic() - start
    memo2 = result2.pydantic

    _print_memo(
        "Kickoff #2 (follow-up query, prior context injected via custom_memory)", memo2
    )
    _print_stats("Kickoff #2", elapsed2, result2.token_usage)

    if memo2 is not None:
        save_memo(COMPANY_NAME, memo2)
        print(f"\nUpdated the custom-memory record for {COMPANY_NAME!r} with kickoff #2's memo.")

    # --- Verification: did kickoff #2's own output carry forward the
    # figures from the injected prior context, or hedge/ignore it anyway?
    # Recall here is deterministic by construction (the figures are
    # literally present in company_info_2's text) -- this checks that the
    # crew actually used what it was given, not whether recall "worked" in
    # the probabilistic sense the old CrewAI-memory version of this test
    # checked. ---
    blob2 = _memo_text_blob(memo2)
    found_markers = [m for m in _RECALL_MARKERS if m.lower() in blob2]
    found_hedges = [h for h in _HEDGE_PHRASES if h in blob2]

    print("\n" + "=" * 70)
    print("CONTEXT-INJECTION CHECK (custom_memory.py)")
    print("=" * 70)
    print(
        f"Kickoff #1 figures found verbatim in kickoff #2's own output "
        f"(present in the injected prior-context block, not re-supplied in "
        f"the follow-up query text itself): {found_markers or 'NONE'}"
    )
    print(f"Hedging/no-information phrases found in kickoff #2's output: {found_hedges or 'none'}")
    if found_markers:
        print(
            "RESULT: kickoff #2 correctly used the injected prior context in "
            "its own output."
        )
    elif found_hedges:
        print(
            "RESULT: kickoff #2 hedged / asked for information despite the "
            "prior context being present in its input -- reported honestly "
            "rather than assumed to have worked."
        )
    else:
        print(
            "RESULT: inconclusive from these markers alone -- inspect "
            "kickoff #2's full memo text above manually."
        )


if __name__ == "__main__":
    run()
