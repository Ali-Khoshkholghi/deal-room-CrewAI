# deal-room

A CrewAI pipeline that turns unstructured pitch text into a decision-ready
investment memo. A Managing Partner agent acts as the crew's **manager**,
deciding at runtime which of its 4 specialists (financial, market, technical,
risk) to consult, in what order, and whether to follow up — all running on
Cerebras via LiteLLM.

---

## Milestone 2: Sequential Crew

`Process.sequential` with four analyst tasks chained via CrewAI's `context=`
mechanism. Each upstream `Task`'s `TaskOutput` is serialized into the
downstream task's prompt.

**Key gotcha:** an *unset* `context` in `Process.sequential` is not "no
context" — it's "auto-aggregate every prior task's output so far." The four
analyst tasks needed `context=[]` explicitly to run independently; without it,
`analyze_market` silently inherited `analyze_financials`'s findings.

---

## Milestone 3: Hierarchical Crew

`Process.hierarchical` — the Managing Partner is passed as `manager_agent=`,
not in `agents`. There is exactly one `Task` (`produce_investment_memo`) with
**no `agent` set**, which is what grants the manager the delegation tool pair
scoped to all crew members. Sequencing of specialist calls is an LLM decision
made fresh on every `kickoff()`.

A structural constraint: the manager cannot also be a regular crew member
(`ValidationError: manager_agent_in_agents`). CrewAI's `@agent` decorator
auto-registers every decorated method into `self.agents`, so `managing_partner()`
in `crew.py` is a plain, undecorated method.

### M2 vs M3

| | M2 — sequential | M3 — hierarchical |
|---|---|---|
| Process | `Process.sequential` | `Process.hierarchical` |
| Who calls each specialist | Fixed order | Manager decides at runtime |
| Specialist coverage | Guaranteed — all 4 always execute | Not guaranteed; 1 of 4 single runs consulted only 2 |
| `output_pydantic` reliability | 100% | ~50% — two distinct failure modes (nested dict in str field; exact Literal value mismatch) |
| Cost (normal pitch) | 60,580 tokens / 25 req / 63.6s | 94,920 tokens / 15 req / 444.4s |
| When to choose | Predictable cost/latency, fixed checklist | Steps vary by input; flexibility worth the reliability cost |

### Re-query behavior: negative result

Across every cleanly-logged run, **the manager never re-queried a specialist
for a thin or hedged answer**, despite the task description explicitly
instructing it to. This was tested deliberately: one pitch stripped financials
to produce a thin answer; the manager accepted it without follow-up every time.
Stated as a finding about this version of CrewAI's hierarchical process, not a
bug to fix.

### `Literal` constraints

`confidence`, `moat_credibility`, `overall_risk_level`, and `recommendation`
are typed as `Literal[...]` in `models.py` — turns silent semantic drift into a
loud validation failure. A `field_validator` normalizes common mismatches
(underscores vs spaces, case) before rejecting genuine garbage.

---

## Milestone 4: Memory and Custom Tools

### Custom memory layer (replaces CrewAI automatic memory)

CrewAI's automatic per-agent-step `Memory` is **fully retired**. Its hooks fire
once per agent step — cost scales with delegation count, not with facts that
actually need to persist — and the save side has no config field to reduce it.
Every lever tried (leaner recall config, decoupled model, manager-only scoping)
reduced cost at the price of losing facts between kickoffs.

`custom_memory.py` exposes two functions, both **zero-LLM-call**:

- **`save_memo(company_name, memo)`** — called once after `kickoff()`. Pulls
  fields straight from the typed `InvestmentMemo` and writes them to LanceDB
  with a local ONNX MiniLM-L6-v2 embedding. A second save for the same company
  overwrites rather than duplicating.
- **`recall_memo(company_name)`** — called once before `kickoff()`. Exact
  metadata match on `company_name`, returns a plain `dict` or `None`.

`main.py`'s two-kickoff flow explicitly recalls the first memo and injects it
as plain-text prior context into the follow-up query — deterministic context
injection instead of probabilistic automatic recall.

**Verification:** zero-cost unit test (hand-built memo, exact round-trip) + one
live kickoff (real memo, save→recall field match). Memory added **zero** extra
requests/tokens to the run.

### Custom tools

- **`WebSearchTool`** (`market_analyst` only): wraps `SerperDevTool` when
  `SERPER_API_KEY` is valid, falls back to keyless `ddgs` otherwise.
- **`PitchDeckReaderTool`**: extracts text from a PDF via PyMuPDF. Attached
  conditionally on `deck_path` (see Milestone 5's Fix 1). The sample deck
  (`brightledger_pitch_deck.pdf`) includes detail `company_info` omits —
  named competitors, tech stack, funding terms — giving specialists a genuine
  informational reason to read it.

**Live verified:** deck-only facts (Tipalti, Bill.com, Kubernetes, SOC 2 Type I,
"$2.5M seed at a $12M pre-money") appeared verbatim in the final memo.

### Cost comparison: M2 → M3 → M4

| | M2 — sequential | M3 — hierarchical | M4 (custom memory) |
|---|---|---|---|
| Tokens | 60,580 | 94,920 | 100,819 crew + **0** memory |
| Requests | 25 | 15 | 23 crew + **0** memory |
| Wall-clock | 63.6s | 444.4s | 498.3s |

M4's memory line item is now zero — no LLM calls, no tokens, no quota consumed.

---

## Milestone 5: CrewAI Flows

`flow.py`'s `DealRoomFlow` wraps the M3/M4 hierarchical crew in a CrewAI
`Flow`, adding a conditional second pass above the crew itself.

**Topology:**

```
run_initial_analysis (@start)
        |
        v
decide_deep_dive (@router — reads state.initial_memo)
        |
   +----+----+
   |         |
"deep_dive"  "skip_deep_dive"
   |         |
   v         v
run_deep_dive   finalize_without_deep_dive
```

Router condition: `memo.confidence == "low" or memo.recommendation == "needs
more diligence"`. A failed parse (`initial_memo is None`) routes to
`skip_deep_dive` rather than crashing.

**State schema** (`DealRoomFlowState`): `company_info`, `pitch_deck_path`,
`initial_memo`, `deep_dive_triggered`, `final_memo`, `pass_count`,
`needs_human_review`. Subclasses `FlowState` — a plain `BaseModel` fails with
a real `ValidationError` on construction because `Flow` requires an `id` field.

**Live test results (2026-08-07):** Router mechanics confirmed correct — reads
the real memo and applies its condition. The `skip_deep_dive` path is
structurally verified but not exercised by a real memo (both test pitches came
back `"needs more diligence"/"medium"`). The deep-dive pass is a deliberate
stub.

**Real gotcha — Flow subclassing:** `Flow`'s method discovery iterates only
`flow_class.__dict__`, not the MRO. Subclassing `DealRoomFlow` to override one
method silently drops the other three from the flow definition, and `kickoff()`
executes nothing. Fix: monkey-patch directly on the parent class instead of
subclassing.

**Fixes (2026-08-08):**

- **`PitchDeckReaderTool` is now conditional.** Previously always attached to
  every specialist with a hardcoded default path — invoked reflexively even
  when no deck was referenced. `DealRoomCrew` now takes an explicit `deck_path:
  str | None = None`; specialists only get the tool when it's set. `flow.py`'s
  `pitch_deck_path` state field now flows through to `DealRoomCrew(deck_path=...)`.

- **Hard cap on escalation.** `DealRoomFlowState` tracks `pass_count` and
  `needs_human_review`. `MAX_PASSES = 2` is enforced inside `decide_deep_dive`
  — defensive but explicit. If the deep-dive result is still uncertain,
  `needs_human_review` is set to `True` and the memo is returned flagged, not
  silently presented as resolved.

---

## Running it

1. **Python 3.11**, create a venv, install dependencies:

   ```bash
   cd deal_room
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set up `.env`:**

   ```bash
   cp .env.example .env
   # set CEREBRAS_API_KEY=your-key-here
   ```

   `SERPER_API_KEY` is optional — `WebSearchTool` falls back to keyless `ddgs`.
   The ONNX embedder downloads ~80MB on first use, cached under
   `~/.cache/chroma/onnx_models/`.

3. **Run:**

   ```bash
   # Main pipeline (two kickoffs with memory recall)
   PYTHONPATH=src python -m deal_room.main 2>&1 | tee /tmp/run.log

   # Flow-based entry point (M5)
   PYTHONPATH=src python -m deal_room.flow_main

   # M3 delegation behavior test
   PYTHONPATH=src python -m deal_room.test_hierarchical

   # M4 integration test (memory + tools + cost)
   PYTHONPATH=src python -m deal_room.verify_m4_integration
   ```

---

## Troubleshooting

**`LLM Provider NOT provided`** — model string lost its `cerebras/` prefix.
Ensure `LLM(model="cerebras/...")` is passed explicitly via `llm=` to each
`Agent`; don't rely on env-var inference.

**`CrewAI demands OPENAI_API_KEY`** — set `OPENAI_API_KEY=NA` as a dummy; this
is a documented CrewAI quirk, not a real OpenAI dependency.

**`ValidationError: manager_agent_in_agents`** — `managing_partner()` was
decorated with `@agent`, which auto-registers it into `self.agents`. Keep it a
plain, undecorated method.

**`Error executing tool. coworker mentioned not found`** — manager's `coworker`
argument didn't match any agent role string. CrewAI's fuzzy matcher usually
self-corrects on the next call. Check `agents.yaml` role strings for unusual
punctuation.

**`429 Tokens per minute limit exceeded`** / **`LLMContextLengthExceededError`**
— real Cerebras rate limit (CrewAI's error label is misleading). `crew.py`'s
`kickoff_with_retry()` retries with a cooldown for transient per-minute limits.

**`429 Tokens per day limit exceeded`** — daily quota, shared at the Cerebras
**account** level (not per key). Retrying with a cooldown won't help; wait for
the daily reset or use a key from a different account.

**`OpenAI API call failed: ...` on a Cerebras run** — not an actual OpenAI
call. `OpenAICompatibleCompletion` (what Cerebras models resolve to) subclasses
`OpenAICompletion`, inheriting its hardcoded error string for any failure.
Check the raw log for `api.cerebras.ai` vs `api.openai.com` in the request URL
to confirm which server was actually called.
