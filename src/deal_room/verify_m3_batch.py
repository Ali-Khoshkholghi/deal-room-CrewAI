"""Milestone 3 small parallel batch verification: n=4 (2 normal-pitch + 2
vague-financials hierarchical crew runs), run CONCURRENTLY via a thread
pool so total wall-clock stays close to one run's duration instead of 4x
that.

This is deliberately a small, quick, directional batch — NOT a
statistically robust sample. A proper n=20 sequential batch was estimated
at 2-3 hours; this trades sample size for a result in ~1 run's worth of
wall-clock time. Treat percentages here as "which way does this lean,"
not as reliable rates.
"""

import csv
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

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

VAGUE_FINANCIALS_PITCH = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched a bit over
a year ago and have been growing steadily. We have 11 people on the team
(7 engineering, 2 sales, 2 ops) and previously raised a pre-seed round
from angel investors and a regional fund. We're now in early conversations
for a seed round.
"""

RUNS_PER_CASE = 2  # 2 normal + 2 vague = 4 total, run concurrently
MAX_WORKERS = 4
CSV_PATH = "/tmp/m3_batch_results.csv"
CSV_FIELDS = [
    "run_id",
    "input_type",
    "specialists_consulted",
    "pydantic_parse_success",
    "error_message",
    "total_tokens",
    "total_requests",
    "wall_clock_seconds",
    "final_recommendation",
    "final_confidence",
    "requery_count",
]

_real_stdout = sys.stdout


def summarize_specialist_calls(calls: list[str]) -> dict:
    """Turn a run's list of delegated-to specialist roles (one entry per
    completed delegation, via DealRoomCrew.pop_specialist_calls()) into the
    same two figures the old log-scraping `analyze_log()` produced:
    distinct specialists consulted, and a re-query count (extra
    delegations to a role beyond its first). This is exact, not a text
    heuristic — it comes straight from step_callback firing on each
    specialist's own AgentFinish, so it isn't subject to the "CrewAI
    doesn't render verbose output on the calling thread under concurrency"
    failure mode that made the log-scraping version return 0 for every
    concurrent run.
    """
    counts = Counter(calls)
    requery_count = sum(count - 1 for count in counts.values() if count > 1)
    return {
        "specialists_consulted": len(counts),
        "requery_count": requery_count,
    }


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "rate_limit", "too many requests")
    )


def run_one(run_id: int, input_type: str, company_info: str) -> dict:
    print(f"[run {run_id:02d} {input_type}] starting", file=_real_stdout)

    row = {
        "run_id": run_id,
        "input_type": input_type,
        "specialists_consulted": 0,
        "pydantic_parse_success": False,
        "error_message": "",
        "total_tokens": 0,
        "total_requests": 0,
        "wall_clock_seconds": 0.0,
        "final_recommendation": "",
        "final_confidence": "",
        "requery_count": 0,
    }

    deal_room_crew = None
    crew = None
    try:
        deal_room_crew = DealRoomCrew()
        crew = deal_room_crew.crew()
        start = time.monotonic()
        result = crew.kickoff(inputs={"company_info": company_info})
        elapsed = time.monotonic() - start

        row["wall_clock_seconds"] = round(elapsed, 1)
        row["total_tokens"] = result.token_usage.total_tokens
        row["total_requests"] = result.token_usage.successful_requests

        memo = result.pydantic
        if memo is not None:
            row["pydantic_parse_success"] = True
            row["final_recommendation"] = memo.recommendation
            row["final_confidence"] = memo.confidence
        else:
            row["error_message"] = "kickoff succeeded but result.pydantic is None"
    except Exception as exc:  # noqa: BLE001 - batch harness needs to survive any failure
        if _is_rate_limit_error(exc):
            row["error_message"] = f"rate_limited: {exc}"[:500]
        else:
            row["error_message"] = f"{type(exc).__name__}: {exc}"[:500]
        if crew is not None:
            try:
                usage = crew.calculate_usage_metrics()
                row["total_tokens"] = usage.total_tokens
                row["total_requests"] = usage.successful_requests
            except Exception:
                pass

    if deal_room_crew is not None:
        calls = deal_room_crew.pop_specialist_calls()
        analysis = summarize_specialist_calls(calls)
        row["specialists_consulted"] = analysis["specialists_consulted"]
        row["requery_count"] = analysis["requery_count"]

    print(
        f"[run {run_id:02d} {input_type}] done in "
        f"{row['wall_clock_seconds']:.1f}s "
        f"(parsed={row['pydantic_parse_success']})",
        file=_real_stdout,
    )
    return row


def summarize(rows: list[dict]):
    print(f"\n{'=' * 70}", file=_real_stdout)
    print(
        f"SMALL PARALLEL BATCH SUMMARY (n={len(rows)}) — directional only, "
        f"not a statistically robust sample",
        file=_real_stdout,
    )
    print("=" * 70, file=_real_stdout)

    for input_type in ("normal", "vague"):
        subset = [r for r in rows if r["input_type"] == input_type]
        n = len(subset)
        if n == 0:
            continue

        all_four = sum(1 for r in subset if r["specialists_consulted"] == 4)
        parsed_ok = sum(1 for r in subset if r["pydantic_parse_success"])
        tokens = [r["total_tokens"] for r in subset if r["total_tokens"]]
        requests = [r["total_requests"] for r in subset if r["total_requests"]]
        times = [r["wall_clock_seconds"] for r in subset if r["wall_clock_seconds"]]
        requeries = sum(r["requery_count"] for r in subset)

        print(f"\n--- {input_type.upper()} (n={n}) ---", file=_real_stdout)
        print(
            f"Consulted all 4 specialists : {all_four}/{n}", file=_real_stdout
        )
        print(f"output_pydantic parsed OK   : {parsed_ok}/{n}", file=_real_stdout)
        if tokens:
            print(
                f"Tokens   mean/median : {statistics.mean(tokens):.0f} / "
                f"{statistics.median(tokens):.0f}",
                file=_real_stdout,
            )
        if requests:
            print(
                f"Requests mean/median : {statistics.mean(requests):.1f} / "
                f"{statistics.median(requests):.1f}",
                file=_real_stdout,
            )
        if times:
            print(
                f"Wall-clock (s) mean/median : {statistics.mean(times):.1f} / "
                f"{statistics.median(times):.1f}",
                file=_real_stdout,
            )
        print(
            f"Genuine re-query events (excl. mechanical retries): {requeries}",
            file=_real_stdout,
        )

    failures = [r for r in rows if not r["pydantic_parse_success"]]
    n_failures = len(failures)
    failures_with_skip = sum(1 for r in failures if r["specialists_consulted"] < 4)
    print(
        f"\n--- Parse-failure / specialist-skip correlation (n={len(rows)}) ---",
        file=_real_stdout,
    )
    if n_failures == 0:
        print("No parse failures occurred in this batch.", file=_real_stdout)
    else:
        print(
            f"{failures_with_skip} of {n_failures} parse failures had "
            f"specialists_consulted < 4 (reported as a correlation on a "
            f"very small n, not a causal claim).",
            file=_real_stdout,
        )


def run():
    tasks = []
    run_id = 0
    for input_type, company_info in (
        ("normal", NORMAL_PITCH),
        ("vague", VAGUE_FINANCIALS_PITCH),
    ):
        for _ in range(RUNS_PER_CASE):
            run_id += 1
            tasks.append((run_id, input_type, company_info))

    batch_start = time.monotonic()
    rows_by_id = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_one, run_id, input_type, company_info): run_id
            for run_id, input_type, company_info in tasks
        }
        for future in as_completed(futures):
            run_id = futures[future]
            rows_by_id[run_id] = future.result()

    batch_elapsed = time.monotonic() - batch_start
    rows = [rows_by_id[run_id] for run_id, _, _ in tasks]

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summarize(rows)
    print(f"\nTotal batch wall-clock (4 runs, {MAX_WORKERS}-way parallel): "
          f"{batch_elapsed:.1f}s")
    print(f"Full per-run CSV written to {CSV_PATH}")


if __name__ == "__main__":
    run()
