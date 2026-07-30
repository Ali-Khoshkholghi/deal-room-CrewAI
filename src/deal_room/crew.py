import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, output_pydantic, task
from dotenv import load_dotenv

from deal_room.models import FinancialAssessment

load_dotenv()

# Cerebras is not one of CrewAI's natively-wrapped providers, so it's routed
# through LiteLLM via the `cerebras/` model prefix. Building the LLM object
# explicitly (rather than relying on env-var inference) avoids CrewAI
# defaulting to an OpenAI-shaped provider guess.
cerebras_llm = LLM(
    model="cerebras/gpt-oss-120b",  # confirmed against GET /v1/models on
                                    # Cerebras' API as of 2026-07-30 — their
                                    # catalog changes, so re-check at build time
    api_key=os.getenv("CEREBRAS_API_KEY"),
)


@CrewBase
class DealRoomCrew:
    """Milestone 1: single-agent, single-task financial analysis pipeline."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # Registers the FinancialAssessment model under the name referenced by
    # `output_pydantic: FinancialAssessment` in tasks.yaml — CrewAI resolves
    # that string against @output_pydantic-decorated class attributes.
    FinancialAssessment = output_pydantic(FinancialAssessment)

    @agent
    def financial_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            llm=cerebras_llm,
            verbose=True,
        )

    @task
    def analyze_financials(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_financials"],
            output_pydantic=FinancialAssessment,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
