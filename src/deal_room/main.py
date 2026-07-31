import time

from dotenv import load_dotenv

from deal_room.crew import DealRoomCrew

load_dotenv()

# Same test pitch as Milestones 1-2, for continuity across milestones.
COMPANY_INFO = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""


def run():
    crew = DealRoomCrew().crew()

    # Run with `PYTHONPATH=src python -m deal_room.main 2>&1 | tee /tmp/m3_run.log`
    # to capture the manager's delegation activity (visible in the verbose
    # trace as "Delegate work to coworker" / "Ask question to coworker"
    # tool calls) to a log file for later inspection — there's only one
    # Task object now, so `result.tasks_output` no longer gives you a
    # per-specialist breakdown the way Milestone 2 did.
    start = time.monotonic()
    result = crew.kickoff(inputs={"company_info": COMPANY_INFO})
    elapsed = time.monotonic() - start

    memo = result.pydantic

    print("\n" + "=" * 70)
    print("FINAL INVESTMENT MEMO (structured)")
    print("=" * 70)
    print(f"Company summary   : {memo.company_summary}")
    print(f"Financial summary : {memo.financial_summary}")
    print(f"Market summary    : {memo.market_summary}")
    print(f"Technical summary : {memo.technical_summary}")
    print(f"Risk summary      : {memo.risk_summary}")
    print(f"Key red flags     : {memo.key_red_flags}")
    print(f"Recommendation    : {memo.recommendation}")
    print(f"Confidence        : {memo.confidence}")

    print("\n=== Raw pydantic object ===")
    print(repr(memo))

    usage = result.token_usage
    print("\n" + "=" * 70)
    print("EXECUTION STATS")
    print("=" * 70)
    print(f"Wall-clock time      : {elapsed:.1f}s for 1 hierarchical task")
    print(f"Successful requests  : {usage.successful_requests}")
    print(f"Prompt tokens        : {usage.prompt_tokens}")
    print(f"Completion tokens    : {usage.completion_tokens}")
    print(f"Total tokens         : {usage.total_tokens}")
    print(
        "Cost estimate        : not computed — Cerebras pricing isn't in "
        "CrewAI's built-in cost table for this model; check current "
        "per-token rates at https://cloud.cerebras.ai/pricing and multiply "
        "against the token counts above for an actual figure."
    )


if __name__ == "__main__":
    run()
