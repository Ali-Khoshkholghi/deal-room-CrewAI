# deal-room — Milestone 1

A single-agent, single-task CrewAI pipeline that turns unstructured pitch
text into a structured financial assessment, running on Cerebras via
LiteLLM.

## What this milestone demonstrates

- **The `agents.yaml` / `tasks.yaml` config pattern**: agent role/goal/backstory
  and task description/expected_output are defined declaratively in YAML
  (`src/deal_room/config/`), then wired into Python objects in `crew.py`
  using the `@CrewBase` / `@agent` / `@task` / `@crew` decorators. This is
  the pattern later milestones (multi-agent, hierarchical, memory/tools,
  Flow) build on top of.
- **Structured output via `output_pydantic`**: the task doesn't just return
  a blob of text — it returns a typed `FinancialAssessment` object
  (`src/deal_room/models.py`) with `burn_rate_assessment`, `runway_estimate`,
  `unit_economics_notes`, `red_flags`, and `confidence` fields. `main.py`
  prints the parsed pydantic object, not raw LLM output, to prove the
  parsing actually worked.
- **Cerebras as a non-native LiteLLM provider**: CrewAI has a handful of
  natively-wrapped providers (OpenAI, Anthropic, Gemini, Azure, Bedrock).
  Cerebras isn't one of them, so it's routed through LiteLLM using the
  `cerebras/<model>` string prefix. `crew.py` builds the `LLM` object
  explicitly and passes it to the agent (`llm=cerebras_llm`) instead of
  relying on env-var inference, which is important — see Troubleshooting
  below.

## Running it

1. **Python version**: this project targets Python 3.11.

2. **Create a venv and install dependencies**:

   ```bash
   cd deal_room
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up your `.env`**:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set:

   ```
   CEREBRAS_API_KEY=your-key-here
   ```

   Get a key from the [Cerebras Cloud dashboard](https://cloud.cerebras.ai/)
   (free tier available). `SERPER_API_KEY` isn't used by Milestone 1 — it's
   a placeholder for a later milestone that adds web-search tooling.

4. **Run the pipeline** (from the `deal_room/` directory):

   ```bash
   PYTHONPATH=src python -m deal_room.main
   ```

   This kicks off the crew against a hardcoded fictional startup pitch and
   prints the resulting `FinancialAssessment` fields plus the raw pydantic
   `repr()`.

## Troubleshooting

**`litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM
provider you are trying to call.`**

This means the model string lost its `cerebras/` prefix somewhere before
reaching LiteLLM — a known issue class with several of CrewAI's non-native
providers (it's been reported with Gemini too). Check that:

- The `LLM` object in `crew.py` still has `model="cerebras/..."` (not just
  `"gpt-oss-120b"`).
- That `LLM` instance is passed directly to the `Agent` via `llm=`, not
  inferred from an `OPENAI_MODEL_NAME` env var — that variable only works
  for OpenAI-shaped defaults.
- You haven't set `OPENAI_MODEL_NAME` anywhere in your shell environment
  that could shadow the explicit LLM object.

**CrewAI demands `OPENAI_API_KEY` even though you're not using OpenAI**

With an explicit `LLM` object this shouldn't happen, but if some internal
validation still complains, set a dummy value rather than a real OpenAI
key:

```
OPENAI_API_KEY=NA
```

This is a documented CrewAI quirk, not a real dependency on OpenAI.

## Milestone roadmap

- **Milestone 1** (this one): single agent, single task, structured
  `output_pydantic` result, Cerebras via LiteLLM.
- **Milestone 2**: sequential crew with 4 analysts (e.g. financial, market,
  team, product) each producing structured output, combined into a
  composite deal memo.
- **Milestone 3**: hierarchical crew with a manager agent delegating to the
  4 analysts and synthesizing their output (`Process.hierarchical`).
- **Milestone 4**: add CrewAI memory (short-term/long-term) and custom
  tools (e.g. web search, document retrieval over a real data room).
- **Milestone 5**: wrap the crew in a CrewAI `Flow` for more control over
  branching, state, and multi-step orchestration beyond a single
  `kickoff()` call.
