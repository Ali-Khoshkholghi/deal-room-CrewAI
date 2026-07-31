import time

from dotenv import load_dotenv

from deal_room.crew import RATE_LIMIT_COOLDOWN_SECONDS, DealRoomCrew, kickoff_with_retry

load_dotenv()

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

# Milestone 4, Part A: a follow-up question about the SAME company
# (referenced by name only), deliberately NOT re-supplying any of the
# figures above -- MRR, churn, team size, funding amount. If CrewAI's
# unified memory (enabled in crew.py's `_new_memory()`) actually recalls
# what the first kickoff learned about Brightledger, the manager/specialists
# should be able to answer with the real figures instead of asking for them
# again or hedging with "insufficient data."
FOLLOW_UP_QUERY = """
Following up on Brightledger, the company we discussed earlier -- do not
ask for its founding details, funding amount, team size, or MRR again,
you already have those from before. Just answer: given what you already
know about Brightledger, is the risk profile still "medium", or has
anything changed? Also state what you recall about its MRR and monthly
churn figures from before, if anything.
"""

# Substrings from COMPANY_INFO that never appear in FOLLOW_UP_QUERY. Their
# presence anywhere in the second run's own output is direct evidence the
# manager/specialists pulled them from memory rather than being re-told.
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


def _print_stats(label: str, elapsed: float, usage, memory_usage) -> None:
    print("\n" + "=" * 70)
    print(f"EXECUTION STATS — {label}")
    print("=" * 70)
    print(f"Wall-clock time      : {elapsed:.1f}s")
    print(f"Successful requests  : {usage.successful_requests}")
    print(f"Prompt tokens        : {usage.prompt_tokens}")
    print(f"Completion tokens    : {usage.completion_tokens}")
    print(f"Total tokens         : {usage.total_tokens}")
    if memory_usage is not None:
        print(
            "Memory analysis LLM  : "
            f"{memory_usage.total_tokens} tokens / "
            f"{memory_usage.successful_requests} requests "
            "(NOT included in the totals above -- separate dedicated "
            "instance, see crew.py's pop_memory_llm_usage())"
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


def run():
    # --- Kickoff #1: full company info (establishes what memory should
    # have to work with for kickoff #2). ---
    crew1 = DealRoomCrew()
    crew_obj_1 = crew1.crew()
    start = time.monotonic()
    result1 = kickoff_with_retry(crew_obj_1, {"company_info": COMPANY_INFO})
    elapsed1 = time.monotonic() - start
    memo1 = result1.pydantic

    _print_memo("Kickoff #1 (Brightledger, full info)", memo1)
    _print_stats(
        "Kickoff #1", elapsed1, result1.token_usage, crew1.pop_memory_llm_usage()
    )

    print(
        f"\nCooling down {RATE_LIMIT_COOLDOWN_SECONDS}s before kickoff #2 to let "
        "the shared Cerebras per-minute quota recover (memory's own LLM calls "
        "add real load on top of the crew's -- see note above run())."
    )
    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)

    # --- Kickoff #2: a NEW DealRoomCrew() instance (fresh LLMs per M3's
    # Fix 1 -- see README), same follow-up question, NO company_info
    # figures re-supplied. Whether this recalls anything is what we're
    # actually testing here -- not assumed. ---
    crew2 = DealRoomCrew()
    crew_obj_2 = crew2.crew()
    start = time.monotonic()
    result2 = kickoff_with_retry(crew_obj_2, {"company_info": FOLLOW_UP_QUERY})
    elapsed2 = time.monotonic() - start
    memo2 = result2.pydantic

    _print_memo("Kickoff #2 (follow-up query, no company_info re-supplied)", memo2)
    _print_stats(
        "Kickoff #2", elapsed2, result2.token_usage, crew2.pop_memory_llm_usage()
    )

    # --- Part A verification: did kickoff #2 show evidence of recall? ---
    blob2 = _memo_text_blob(memo2)
    found_markers = [m for m in _RECALL_MARKERS if m.lower() in blob2]
    found_hedges = [h for h in _HEDGE_PHRASES if h in blob2]

    print("\n" + "=" * 70)
    print("MEMORY RECALL CHECK (Milestone 4, Part A)")
    print("=" * 70)
    print(
        f"Kickoff #1 figures found verbatim in kickoff #2's own output "
        f"(never re-supplied in kickoff #2's input): {found_markers or 'NONE'}"
    )
    print(f"Hedging/no-information phrases found in kickoff #2's output: {found_hedges or 'none'}")
    if found_markers:
        print(
            "RESULT: kickoff #2 reproduced specific figures from kickoff #1 "
            "without being re-told them -- real evidence of recall, not "
            "an assumption."
        )
    elif found_hedges:
        print(
            "RESULT: kickoff #2 hedged / asked for information instead of "
            "recalling it -- memory did NOT visibly help here, reported "
            "honestly rather than assumed to have worked."
        )
    else:
        print(
            "RESULT: inconclusive from these markers alone -- inspect "
            "kickoff #2's full memo text above manually."
        )


if __name__ == "__main__":
    run()
