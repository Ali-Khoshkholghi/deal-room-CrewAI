"""Milestone 4, Part C: integration test for the M3 hierarchical crew with
memory (Part A) and both custom tools (Part B) attached.

Runs a single kickoff with the normal-pitch company_info plus a reference
to the sample pitch deck PDF, and verifies -- via CrewAI's own
`ToolUsageFinishedEvent` on the global event bus, not log-scraping (see
README's M3 "Known measurement limitations" for why log-scraping was
abandoned) -- that WebSearchTool and PitchDeckReaderTool were each
actually invoked at least once, not just available.
"""

import time

from dotenv import load_dotenv

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

from deal_room.crew import DealRoomCrew, kickoff_with_retry

load_dotenv()

COMPANY_INFO = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round. A pitch deck PDF with more detail on our
architecture, named competitors, and funding terms is available to any
specialist who wants it via their pitch deck reading tool.
"""

TRACKED_TOOLS = {"web_search", "read_pitch_deck"}

# M3 baseline for comparison (from README's "Known measurement limitations,
# and the fix" re-run section): a single clean, uncontaminated normal-pitch
# run with the M3 Fix 1/2/3 changes in place, no memory, no extra tools.
M3_BASELINE_TOKENS = 94_920
M3_BASELINE_REQUESTS = 15
M3_BASELINE_SECONDS = 444.4

_tool_calls: list[tuple[str, str]] = []


@crewai_event_bus.on(ToolUsageFinishedEvent)
def _on_tool_usage_finished(source, event: ToolUsageFinishedEvent) -> None:
    if event.tool_name in TRACKED_TOOLS:
        output_snippet = str(event.output)[:200]
        _tool_calls.append((event.tool_name, output_snippet))


def run():
    deal_room_crew = DealRoomCrew()
    crew = deal_room_crew.crew()

    start = time.monotonic()
    result = kickoff_with_retry(crew, {"company_info": COMPANY_INFO})
    elapsed = time.monotonic() - start

    crewai_event_bus.flush()

    memo = result.pydantic
    usage = result.token_usage
    memory_usage = deal_room_crew.pop_memory_llm_usage()
    specialist_calls = deal_room_crew.pop_specialist_calls()

    print("\n" + "=" * 70)
    print("MILESTONE 4 INTEGRATION TEST RESULTS")
    print("=" * 70)

    print("\n--- output_pydantic parsing ---")
    if memo is not None:
        print("PASSED: result.pydantic parsed as a valid InvestmentMemo")
        print(f"  recommendation: {memo.recommendation}")
        print(f"  confidence: {memo.confidence}")
    else:
        print("FAILED: result.pydantic is None")

    print("\n--- Tool invocation check (via ToolUsageFinishedEvent, not log-scraping) ---")
    called_tool_names = {name for name, _ in _tool_calls}
    for tool_name in sorted(TRACKED_TOOLS):
        count = sum(1 for name, _ in _tool_calls if name == tool_name)
        status = "INVOKED" if count > 0 else "NOT INVOKED"
        print(f"  {tool_name}: {status} ({count}x)")
    for tool_name, snippet in _tool_calls:
        print(f"    [{tool_name}] output snippet: {snippet!r}")

    missing = TRACKED_TOOLS - called_tool_names
    if missing:
        print(f"\nWARNING: these tools were never invoked in this run: {missing}")
    else:
        print("\nBoth custom tools were invoked at least once in this run.")

    print("\n--- specialists_consulted (Fix 3 mechanism, reused here) ---")
    print(f"  raw calls: {specialist_calls}")

    print("\n--- Cost vs. M3 baseline ---")
    print(f"{'':20s} {'M3 baseline':>15s} {'M4 (this run)':>15s}")
    print(f"{'Total tokens':20s} {M3_BASELINE_TOKENS:>15,d} {usage.total_tokens:>15,d}")
    print(f"{'Requests':20s} {M3_BASELINE_REQUESTS:>15d} {usage.successful_requests:>15d}")
    print(f"{'Wall-clock (s)':20s} {M3_BASELINE_SECONDS:>15.1f} {elapsed:>15.1f}")
    token_delta = usage.total_tokens - M3_BASELINE_TOKENS
    token_pct = (token_delta / M3_BASELINE_TOKENS) * 100
    print(f"\nToken delta vs. M3 baseline: {token_delta:+,d} ({token_pct:+.1f}%)")
    if memory_usage is not None:
        print(
            f"Memory analysis LLM (separate instance, NOT in the totals "
            f"above): {memory_usage.total_tokens} tokens / "
            f"{memory_usage.successful_requests} requests"
        )
        print(
            f"True total cost including memory overhead: "
            f"{usage.total_tokens + memory_usage.total_tokens:,d} tokens"
        )


if __name__ == "__main__":
    run()
