"""Milestone 4 custom tools: real web search and PDF pitch-deck reading."""

import logging
import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The web search query to run.")


class WebSearchTool(BaseTool):
    """Real web search for market/competitive research, so the market
    analyst isn't limited to whatever the pitch text itself says.

    Wraps crewai-tools' `SerperDevTool` (Serper.dev) when `SERPER_API_KEY`
    is configured. Falls back to `ddgs` (a keyless DuckDuckGo search
    wrapper) when Serper isn't usable -- verified directly in this
    environment: the `SERPER_API_KEY` present in `.env` returns `403
    Forbidden` (an unauthorized/placeholder key -- the README already noted
    Serper "isn't used yet"). A hand-rolled `requests`+HTML-scrape fallback
    was tried first and rejected: DuckDuckGo's plain HTML endpoint
    intermittently returned a bot-detection challenge page instead of
    results, even moments after a successful call, which would make this
    tool unreliable for the exact thing Part C needs to verify (a real
    invocation with real results). `ddgs` handles that anti-bot handling
    itself and returned results reliably in testing.

    The fallback is automatic and per-call: a Serper failure mid-run still
    falls through to `ddgs` for that query rather than failing outright.
    """

    name: str = "web_search"
    description: str = (
        "Search the live web for real, current information -- named "
        "competitors, market-size figures, industry news, pricing pages -- "
        "instead of relying only on the text provided about the company. "
        "Provide a 'query' string."
    )
    args_schema: type[BaseModel] = WebSearchInput

    _serper: Any = PrivateAttr(default=None)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if os.getenv("SERPER_API_KEY"):
            try:
                from crewai_tools import SerperDevTool

                self._serper = SerperDevTool()
            except Exception as e:  # noqa: BLE001 - degrade to the fallback, don't crash tool construction
                logger.warning("SerperDevTool unavailable (%s); using ddgs fallback.", e)

    def _run(self, query: str) -> str:
        if self._serper is not None:
            try:
                return str(self._serper.run(search_query=query))
            except Exception as e:  # noqa: BLE001 - fall through to the keyless backend
                logger.warning(
                    "Serper search failed (%s); falling back to ddgs for this query.", e
                )
        return self._ddgs_search(query)

    def _ddgs_search(self, query: str, max_results: int = 5) -> str:
        try:
            from ddgs import DDGS

            results = DDGS().text(query, max_results=max_results)
        except Exception as e:  # noqa: BLE001 - report the failure back to the agent
            return f"Web search failed: {e}"

        if not results:
            return f"No web search results found for '{query}'."

        lines = [
            f"- {r.get('title', '')} ({r.get('href', '')}): {r.get('body', '')}"
            for r in results
        ]
        return f"Web search results for '{query}':\n" + "\n".join(lines)


class PitchDeckReaderInput(BaseModel):
    file_path: str | None = Field(
        default=None,
        description=(
            "Path to the pitch deck PDF to read. Omit to read the default "
            "deck configured for this run, if one is set."
        ),
    )


class PitchDeckReaderTool(BaseTool):
    """Extracts text from a PDF pitch deck via PyMuPDF, so a specialist can
    pull detail beyond what's in the plain-text company_info summary.

    A `default_file_path` set at construction is read when the LLM omits
    `file_path` at call time -- mirrors crewai-tools' own `FileReadTool`
    pattern for a pre-bound default vs. a runtime override.
    """

    name: str = "read_pitch_deck"
    description: str = "Extract the text content of a pitch deck PDF."
    args_schema: type[BaseModel] = PitchDeckReaderInput
    default_file_path: str | None = None

    def __init__(self, default_file_path: str | None = None, **kwargs: Any) -> None:
        if default_file_path and "description" not in kwargs:
            kwargs["description"] = (
                "Extract the text content of the company's pitch deck PDF -- "
                "use it for detail beyond the plain-text company summary "
                f"(architecture, named competitors, funding terms, etc). "
                f"A deck is available at '{default_file_path}'; call this "
                "tool with no arguments to read it, or pass a different "
                "'file_path' to read another PDF."
            )
        super().__init__(default_file_path=default_file_path, **kwargs)

    def _run(self, file_path: str | None = None) -> str:
        target = file_path or self.default_file_path
        if not target:
            return "Error: no file_path provided and no default pitch deck configured."

        path = Path(target)
        if not path.exists():
            return f"Error: file not found at {target}"

        try:
            doc = fitz.open(str(path))
            pages = [page.get_text() for page in doc]
            doc.close()
        except Exception as e:  # noqa: BLE001 - report the failure back to the agent, don't crash the run
            return f"Error reading PDF at {target}: {e}"

        text = "\n\n".join(p.strip() for p in pages if p.strip())
        if not text:
            return f"No extractable text found in {target} (it may be scanned/image-based)."
        return text
