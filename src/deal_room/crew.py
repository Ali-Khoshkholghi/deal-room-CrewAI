import os
import threading
import time
import uuid
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import AgentExecutionCompletedEvent
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.unified_memory import Memory
from crewai.project import CrewBase, agent, crew, output_pydantic, task
from dotenv import load_dotenv

from deal_room.generate_sample_pitch_deck import SAMPLE_DECK_PATH
from deal_room.models import (
    FinancialAssessment,
    InvestmentMemo,
    MarketAssessment,
    RiskAssessment,
    TechnicalAssessment,
)
from deal_room.tools import PitchDeckReaderTool, WebSearchTool

load_dotenv()

# Project-local, persistent LanceDB path for the unified memory system
# (Milestone 4, Part A). Explicit rather than CrewAI's global default data
# dir so the memory store this project builds up is self-contained,
# inspectable, and doesn't risk colliding with an embedding-dimension
# mismatch from some unrelated app's data at the shared default path (see
# README M4 section for why that matters: the default vector_dim assumes
# OpenAI's 3072-dim embeddings, ours are ONNX MiniLM's 384-dim).
_MEMORY_STORAGE_PATH = Path(__file__).resolve().parents[2] / ".crewai_memory"


class SpecialistCallLog:
    """Thread-safe registry of which specialist agents were actually
    delegated to, keyed by run_id.

    Replaces the earlier threading.local()-based log-scraping approach for
    `specialists_consulted`, which silently returned 0 under concurrent
    execution because CrewAI doesn't render verbose output on the thread
    that called kickoff().

    This was first attempted with Agent-level `step_callback`, per CrewAI's
    documented step/task callback mechanism — but empirically that callback
    never fires for our specialists at all: CrewAI's `AgentExecutor` has a
    fast path for agents with zero tools
    (`_invoke_loop_native_no_tools` in `crewai.agents.crew_agent_executor`)
    that skips `_invoke_step_callback` entirely, and every specialist here
    has `allow_delegation=False` with no explicit tools, so it always takes
    that path. Verified directly: two live kickoffs with step_callback
    wired returned an empty call list both times despite the delegation
    clearly happening (visible in the verbose trace and in the resulting
    memo's specialist-derived content).

    Instead this hooks `AgentExecutionCompletedEvent` on CrewAI's global
    `crewai_event_bus` — a callback-style hook fired unconditionally at the
    end of every `Agent.execute_task()` call (see
    `crewai.agent.core.Agent._finalize_task_execution`), regardless of
    which internal tool-calling path was taken. It carries the actual
    `Agent` instance that ran, so a run-scoped registry keyed by
    `id(agent)` (registered when each specialist Agent is built) tells us
    which run a given completion belongs to — no log parsing, and no
    dependence on which loop CrewAI happens to pick internally.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, list[str]] = {}
        self._agent_registry: dict[int, tuple[str, str]] = {}

    def start_run(self, run_id: str) -> None:
        with self._lock:
            self._calls[run_id] = []

    def register_agent(self, agent_instance: Agent, run_id: str, role: str) -> None:
        with self._lock:
            self._agent_registry[id(agent_instance)] = (run_id, role)

    def record_completed_execution(self, agent_instance: object) -> None:
        with self._lock:
            entry = self._agent_registry.get(id(agent_instance))
            if entry is None:
                return  # not one of ours (e.g. the manager itself)
            run_id, role = entry
            self._calls.setdefault(run_id, []).append(role)

    def end_run(self, run_id: str) -> list[str]:
        """Pop and return the recorded specialist calls for a run, and
        forget which agent instances belonged to it."""
        with self._lock:
            stale_agent_ids = [
                agent_id
                for agent_id, (owning_run_id, _role) in self._agent_registry.items()
                if owning_run_id == run_id
            ]
            for agent_id in stale_agent_ids:
                del self._agent_registry[agent_id]
            return self._calls.pop(run_id, [])


specialist_call_log = SpecialistCallLog()


@crewai_event_bus.on(AgentExecutionCompletedEvent)
def _on_agent_execution_completed(source: object, event: AgentExecutionCompletedEvent) -> None:
    specialist_call_log.record_completed_execution(event.agent)


@CrewBase
class DealRoomCrew:
    """Milestone 3: hierarchical crew — Managing Partner delegates to 4 specialists."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # tasks.yaml still carries Milestone 2's 4 analyst task definitions
    # (analyze_financials, analyze_market, analyze_technical, assess_risk)
    # as historical reference — this crew's task list (see `crew()` below)
    # no longer builds Task objects from them, so they never execute.
    # Reason: under Process.hierarchical, CrewAI ignores a task's own
    # `agent` field for execution purposes — every task actually runs
    # through `manager_agent` (see crewai/crew.py:
    # `executing_agent = self.manager_agent if is_hierarchical else
    # task.agent`). Keeping those 4 as real crew tasks would just force 4
    # extra manager-mediated passes with no real delegation choice (a task
    # with a fixed `agent` only grants the manager a delegation tool
    # scoped to that one agent). The single `produce_investment_memo` task
    # below, with no `agent` set at all, is what actually grants the
    # manager a delegation tool scoped to *all four* specialists.
    #
    # CrewAI still eagerly resolves every `output_pydantic:` / `agent:`
    # string found anywhere in tasks.yaml at crew-construction time,
    # regardless of whether a Task object ever gets built from that entry
    # — so these registrations (and the 4 @agent methods below) still have
    # to exist, or construction KeyErrors trying to resolve the unused
    # YAML entries.
    FinancialAssessment = output_pydantic(FinancialAssessment)
    MarketAssessment = output_pydantic(MarketAssessment)
    TechnicalAssessment = output_pydantic(TechnicalAssessment)
    RiskAssessment = output_pydantic(RiskAssessment)
    InvestmentMemo = output_pydantic(InvestmentMemo)

    def _new_llm(self) -> LLM:
        """Build a brand-new Cerebras LLM client — called once per agent,
        never cached or shared between agents or across DealRoomCrew
        instances.

        Two distinct bugs are avoided by never sharing an LLM object:

        1. Cross-run contamination: `get_token_usage_summary()` accumulates
           on the LLM instance for its entire lifetime, not per kickoff().
           A single module-level LLM shared across kickoffs (the old
           design) made token/cost figures a running total across runs
           rather than a per-run measurement (see README's M3 "Known
           measurement limitations").
        2. Cross-agent inflation, *within* a single run: `_token_usage` is
           a private attribute on the LLM instance itself
           (`crewai.llms.base_llm.BaseLLM`), and
           `Crew.calculate_usage_metrics()` sums
           `agent.llm.get_token_usage_summary()` over every agent in the
           crew. If multiple agents shared one LLM instance, that same
           cumulative dict would get added into the crew-wide total once
           per agent sharing it — e.g. 5 agents sharing one LLM would
           report 5x the real token count for a single run.

        A fresh, unshared instance per agent (still rebuilt on every new
        DealRoomCrew(), so nothing survives across kickoffs either) keeps
        both dimensions correct: no inflation within a run, no
        contamination across runs.
        """
        return LLM(
            model="cerebras/gpt-oss-120b",  # confirmed against GET
                                            # /v1/models on Cerebras' API as
                                            # of 2026-07-30 — their catalog
                                            # changes, so re-check at build
                                            # time
            api_key=os.getenv("CEREBRAS_API_KEY"),
        )

    @property
    def run_id(self) -> str:
        """Unique id for this DealRoomCrew instance's run, used to key
        `specialist_call_log` so concurrent kickoffs don't cross-contaminate
        each other's specialists_consulted counts."""
        if getattr(self, "_run_id", None) is None:
            self._run_id = str(uuid.uuid4())
            specialist_call_log.start_run(self._run_id)
        return self._run_id

    def _register_specialist(self, agent_instance: Agent, role: str) -> Agent:
        """Register a specialist Agent instance so the global
        AgentExecutionCompletedEvent listener can attribute its completions
        to this run. A coworker-not-found mechanical retry never reaches
        the target agent's `execute_task()` at all — no event fires — so
        only genuine delegations get counted, with no log-scraping needed.
        """
        specialist_call_log.register_agent(agent_instance, self.run_id, role)
        return agent_instance

    def _new_memory(self) -> Memory:
        """Build a fresh unified `Memory` instance (short-term recall +
        entity tracking) for this run, backed by fully local/free
        components — no OpenAI dependency:

        - `embedder={"provider": "onnx"}`: chromadb's built-in ONNX
          all-MiniLM-L6-v2 model, downloaded once (~80MB) and cached
          locally, run entirely on-device. CrewAI's actual default here
          (passing no `embedder=` at all) is `OpenAIEmbeddingFunction` with
          `text-embedding-3-small` — its own default-embedder docstring
          claims ONNX MiniLM but the code builds OpenAI regardless — so
          leaving this unset would have silently required OPENAI_API_KEY.
        - `llm=self._new_llm()`: a dedicated fresh Cerebras instance for
          Memory's own internal encode/recall analysis, instead of the
          class default `llm="gpt-5.4-mini"` (also OpenAI-shaped). Kept
          separate from any agent's own LLM instance — same reasoning as
          `_new_llm()`'s own docstring: no shared instances, no
          contamination.
        - `storage=LanceDBStorage(path=_MEMORY_STORAGE_PATH)`: local,
          file-based, no server or API key. Deliberately a fixed
          project-local path (not per-instance/per-run) so that memories
          saved in one kickoff persist and are recallable by a *different*
          `DealRoomCrew()` instance in a later kickoff — this is what
          Part A's two-call recall test depends on.

        Verified directly (see README M4 section) that this configuration
        remembers and recalls correctly with zero OpenAI calls involved.
        """
        self._memory_llm_instance = self._new_llm()
        return Memory(
            llm=self._memory_llm_instance,
            embedder={"provider": "onnx"},
            storage=LanceDBStorage(path=str(_MEMORY_STORAGE_PATH)),
        )

    def pop_memory_llm_usage(self):
        """Return this run's Memory analysis LLM usage (prompt/completion/
        total tokens, request count), or None if memory wasn't built yet.

        Memory's own LLM calls (extracting entities/categories on
        `remember()`, distilling sub-queries on `recall(depth="deep")`)
        run on a dedicated instance never passed to `Crew.agents` or
        `manager_agent` — so `Crew.calculate_usage_metrics()` /
        `result.token_usage` never sees them. This is the only way to
        observe that cost; report it alongside `result.token_usage`,
        don't substitute one for the other.
        """
        if getattr(self, "_memory_llm_instance", None) is None:
            return None
        return self._memory_llm_instance.get_token_usage_summary()

    def pop_specialist_calls(self) -> list[str]:
        """Return (and clear) the list of specialist roles delegated to
        during this instance's run, one entry per completed delegation
        (repeats included, so callers can derive both distinct-specialist
        counts and re-query counts). Call after kickoff().

        `crewai_event_bus.emit()` dispatches sync handlers on a background
        thread pool and returns immediately — so `AgentExecutionCompletedEvent`
        for the last delegation in a run can still be in flight when
        `kickoff()` returns. `flush()` blocks until every pending handler
        (including ours) has actually run, so this never reads the registry
        before the last event lands.
        """
        crewai_event_bus.flush()
        return specialist_call_log.end_run(self.run_id)

    @agent
    def financial_analyst(self) -> Agent:
        return self._register_specialist(
            Agent(
                config=self.agents_config["financial_analyst"],
                llm=self._new_llm(),
                allow_delegation=False,
                verbose=True,
                tools=[PitchDeckReaderTool(default_file_path=str(SAMPLE_DECK_PATH))],
            ),
            "Startup Financial Analyst",
        )

    @agent
    def market_analyst(self) -> Agent:
        return self._register_specialist(
            Agent(
                config=self.agents_config["market_analyst"],
                llm=self._new_llm(),
                allow_delegation=False,
                verbose=True,
                tools=[
                    WebSearchTool(),
                    PitchDeckReaderTool(default_file_path=str(SAMPLE_DECK_PATH)),
                ],
            ),
            "Market & Competitive Analyst",
        )

    @agent
    def technical_diligence_agent(self) -> Agent:
        return self._register_specialist(
            Agent(
                config=self.agents_config["technical_diligence_agent"],
                llm=self._new_llm(),
                allow_delegation=False,
                verbose=True,
                tools=[PitchDeckReaderTool(default_file_path=str(SAMPLE_DECK_PATH))],
            ),
            "Technical Due Diligence Lead",
        )

    @agent
    def risk_assessor(self) -> Agent:
        return self._register_specialist(
            Agent(
                config=self.agents_config["risk_assessor"],
                llm=self._new_llm(),
                allow_delegation=False,
                verbose=True,
                tools=[PitchDeckReaderTool(default_file_path=str(SAMPLE_DECK_PATH))],
            ),
            "Risk & Governance Assessor",
        )

    # Deliberately NOT decorated with @agent. The Managing Partner is the
    # crew's MANAGER (passed as `manager_agent=` in `crew()` below), not
    # one of the workers in `self.agents`. CrewAI's own validator rejects a
    # crew whose manager_agent instance also appears in `agents`
    # ("manager_agent_in_agents") — and the @agent decorator's method
    # scanning would auto-add it there, since it adds every @agent-decorated
    # method to `self.agents` regardless of whether a task references it.
    # This worker/manager structural split has no LangGraph equivalent:
    # LangGraph has no built-in notion of a "manager node" distinct from a
    # "worker node" — any node can conditionally route to any other, so
    # there's no separate object needing its own instantiation path.
    def managing_partner(self) -> Agent:
        return Agent(
            config=self.agents_config["managing_partner"],
            llm=self._new_llm(),
            allow_delegation=True,
            verbose=True,
        )

    @task
    def produce_investment_memo(self) -> Task:
        # No `agent=` here, and none set in tasks.yaml either. That
        # absence is the whole point: with no fixed executor, CrewAI's
        # hierarchical process routes this task to `manager_agent`, who is
        # given "delegate work to coworker" / "ask question to coworker"
        # tools scoped to every agent in `self.agents` (all 4 specialists).
        # The manager decides, at runtime, which specialists to call, in
        # what order, how many times, and what exactly to ask each one —
        # none of that sequencing is encoded anywhere in this codebase.
        #
        # This is the core CrewAI hierarchical pattern with no clean
        # LangGraph equivalent: a LangGraph graph's topology (which node
        # can run next) is fixed at graph-build time, even for conditional
        # edges — the set of possible next steps is author-defined. Here,
        # the sequence and repetition of specialist calls is itself an LLM
        # decision made fresh on every kickoff, not a path chosen from a
        # pre-declared set of edges.
        return Task(
            config=self.tasks_config["produce_investment_memo"],
            output_pydantic=InvestmentMemo,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.managing_partner(),
            memory=self._new_memory(),
            verbose=True,
        )


# Milestone 4: enabling memory adds its own LLM calls on top of the crew's
# own delegation calls -- every agent step triggers a memory save-extraction
# call, and every step's prompt-building triggers a memory recall call
# (CrewAI's default recall depth is "deep", itself LLM-driven). Observed
# directly: a single kickoff with memory enabled made 75 *separate*
# memory-analysis LLM requests on top of the crew's own 22 -- enough load
# against one shared Cerebras API key to exhaust its per-minute token quota
# mid-run, including during CrewAI's own rate-limit-recovery path (which
# itself makes an LLM call to summarize messages), with no further
# fallback -- an unguarded kickoff() crashed outright on this exact path in
# testing. RATE_LIMIT_COOLDOWN_SECONDS/kickoff_with_retry give callers a
# cooldown and small retry budget instead of losing an entire run to a
# transient quota exhaustion.
RATE_LIMIT_COOLDOWN_SECONDS = 90
MAX_KICKOFF_RETRIES = 2


def kickoff_with_retry(crew: Crew, inputs: dict, max_retries: int = MAX_KICKOFF_RETRIES):
    """Run crew.kickoff(), retrying with a cooldown pause on failure.

    See the module-level comment above for why this exists: memory's own
    LLM traffic can exhaust the shared Cerebras rate limit badly enough
    that CrewAI's own internal recovery path fails too, with no fallback
    left, propagating as an unhandled exception.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return crew.kickoff(inputs=inputs)
        except Exception as e:  # noqa: BLE001 - any failure here should trigger backoff+retry
            last_exc = e
            if attempt < max_retries:
                print(
                    f"kickoff failed ({type(e).__name__}: {e}) -- "
                    f"waiting {RATE_LIMIT_COOLDOWN_SECONDS}s and retrying "
                    f"(attempt {attempt + 2}/{max_retries + 1})"
                )
                time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
    assert last_exc is not None
    raise last_exc
