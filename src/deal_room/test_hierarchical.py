"""Milestone 3 verification: does the hierarchical manager actually delegate
dynamically, or does it behave like a fixed pipeline in disguise?

Runs two cases, each logging the full verbose trace (including every
"Delegate work to coworker" / "Ask question to coworker" tool call) to a
file, then reports which specialists were contacted and how many times.
"""

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from deal_room.crew import DealRoomCrew

load_dotenv()

NORMAL_PITCH = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""

# Same shape, but nearly all financial signal stripped out — no MRR, no
# growth, no churn, no funding amount — to see whether the manager notices
# the financial answer will be thin and re-queries financial_analyst for
# more detail, or just accepts a one-line "not enough data" response.
VAGUE_FINANCIALS_PITCH = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched a bit over
a year ago and have been growing steadily. We have 11 people on the team
(7 engineering, 2 sales, 2 ops) and previously raised a pre-seed round
from angel investors and a regional fund. We're now in early conversations
for a seed round.
"""

SPECIALIST_ROLES = [
    "Startup Financial Analyst",
    "Market & Competitive Analyst",
    "Technical Due Diligence Lead",
    "Risk & Governance Assessor",
]


class Tee:
    """Writes to both the real stdout and a log file, so the run stays
    visible in the console while also being captured for later grepping."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def count_delegation_calls(log_text: str) -> dict[str, int]:
    """Rough count of how many times each specialist's role appears as the
    target of a delegate/ask-question tool call in the verbose trace."""
    counts = {}
    for role in SPECIALIST_ROLES:
        # Matches the role name appearing near "coworker"/tool-call context,
        # case-insensitive, tolerant of the trailing newline CrewAI strips
        # internally but that may still show up in printed panels.
        pattern = re.compile(re.escape(role), re.IGNORECASE)
        counts[role] = len(pattern.findall(log_text))
    return counts


def run_case(label: str, company_info: str, log_path: str):
    print(f"\n{'#' * 70}")
    print(f"# RUNNING CASE: {label}")
    print(f"# Logging full verbose trace to: {log_path}")
    print("#" * 70)

    log_file = open(log_path, "w")
    tee_out = Tee(sys.__stdout__, log_file)
    original_stdout = sys.stdout
    sys.stdout = tee_out

    # A hierarchical manager's final answer isn't forced through a single
    # fixed prompt template the way M2's tasks were, so output_pydantic
    # parsing here is observably less reliable — this occasionally raises
    # (e.g. the manager nests structured JSON inside a field the model
    # declares as `str`). Catching it here lets one case's failure surface
    # as a reportable result rather than aborting the whole comparison run.
    error = None
    result = None
    elapsed = None
    try:
        crew = DealRoomCrew().crew()
        start = time.monotonic()
        result = crew.kickoff(inputs={"company_info": company_info})
        elapsed = time.monotonic() - start
    except Exception as exc:  # noqa: BLE001 - intentionally broad for a verification harness
        error = exc
    finally:
        sys.stdout = original_stdout
        log_file.close()

    log_text = Path(log_path).read_text()
    delegation_counts = count_delegation_calls(log_text)

    return {
        "label": label,
        "result": result,
        "error": error,
        "elapsed": elapsed,
        "log_path": log_path,
        "delegation_counts": delegation_counts,
    }


def print_case_summary(case: dict):
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — {case['label']}")
    print("=" * 70)

    if case["error"] is not None:
        print(f"CASE FAILED: {type(case['error']).__name__}: {case['error']}")
        print(f"(Full trace in {case['log_path']} up to the point of failure.)")
        print("Role mentions in verbose trace (rough delegation-frequency proxy):")
        for role, count in case["delegation_counts"].items():
            print(f"  {role:32s}: {count}")
        return

    memo = case["result"].pydantic
    usage = case["result"].token_usage

    print(f"Recommendation       : {memo.recommendation}")
    print(f"Confidence           : {memo.confidence}")
    print(f"Financial summary    : {memo.financial_summary}")
    print(f"Key red flags        : {memo.key_red_flags}")
    print(f"Wall-clock time      : {case['elapsed']:.1f}s")
    print(f"Successful requests  : {usage.successful_requests}")
    print(f"Total tokens         : {usage.total_tokens}")
    print("Role mentions in verbose trace (rough delegation-frequency proxy):")
    for role, count in case["delegation_counts"].items():
        print(f"  {role:32s}: {count}")


def run():
    normal_case = run_case("NORMAL PITCH", NORMAL_PITCH, "/tmp/m3_run.log")
    print_case_summary(normal_case)

    vague_case = run_case(
        "VAGUE-FINANCIALS PITCH", VAGUE_FINANCIALS_PITCH, "/tmp/m3_vague_run.log"
    )
    print_case_summary(vague_case)

    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print("=" * 70)
    for case in (normal_case, vague_case):
        if case["error"] is not None:
            print(f"{case['label']:24s}: FAILED — {case['error']}")
            continue
        usage = case["result"].token_usage
        print(
            f"{case['label']:24s}: {case['elapsed']:.1f}s, "
            f"{usage.successful_requests} requests, {usage.total_tokens} tokens"
        )


if __name__ == "__main__":
    run()
