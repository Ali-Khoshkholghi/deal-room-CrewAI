"""Custom, minimal-call memory layer replacing CrewAI's automatic per-agent-
step `Memory` (see README's "Custom memory layer" section for why: CrewAI's
own unified memory was measured at 245 requests / 440K tokens per kickoff
for bookkeeping alone, and still lost facts under manager-only scoping).

Reuses the storage/embedding infrastructure already verified working
elsewhere in this project -- LanceDB for storage, chromadb's bundled local
ONNX MiniLM-L6-v2 embedder for vectors (`~/.cache/chroma/onnx_models/`,
already downloaded by earlier CrewAI `Memory` usage) -- but drives it
directly instead of through CrewAI's automatic per-step hooks.

`save_memo()` and `recall_memo()` are each meant to be called exactly ONCE
per `kickoff()`, by the caller, not once per internal agent step. Both are
zero-LLM-call by construction:
- `save_memo()` pulls facts straight from the already-typed `InvestmentMemo`
  Pydantic fields -- no LLM extraction needed, since the crew's own
  `output_pydantic` already did that structuring work.
- `recall_memo()` is an exact metadata match on `company_name`, not a
  similarity search -- this is a per-company lookup keyed by name (the
  caller already knows which company it's asking about), not a general
  "find anything relevant" query, so there's no LLM-driven query analysis
  or consolidation step to run.

An embedding is still computed and stored on save (local ONNX, no LLM call)
even though `recall_memo()` doesn't use it -- kept for potential future
fuzzy/semantic lookup, not exercised by this module's current callers.
"""
from __future__ import annotations

from pathlib import Path

import lancedb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

from deal_room.models import InvestmentMemo

# Sibling to crew.py's `.crewai_memory` (CrewAI's own, now-unused-by-default
# storage) -- same project-root-relative convention, own directory so the
# two never collide.
_STORAGE_PATH = Path(__file__).resolve().parents[2] / ".custom_memory"
_TABLE_NAME = "company_memos"

_embedder: ONNXMiniLM_L6_V2 | None = None


def _get_embedder() -> ONNXMiniLM_L6_V2:
    global _embedder
    if _embedder is None:
        _embedder = ONNXMiniLM_L6_V2()
    return _embedder


def _escape(value: str) -> str:
    """Escape single quotes for a LanceDB SQL-style `where()` predicate."""
    return value.replace("'", "''")


def _combined_summary(memo: InvestmentMemo) -> str:
    return " ".join(
        [
            memo.company_summary,
            memo.financial_summary,
            memo.market_summary,
            memo.technical_summary,
            memo.risk_summary,
            " ".join(memo.key_red_flags),
        ]
    )


def save_memo(company_name: str, memo: InvestmentMemo) -> None:
    """Persist `memo`'s structured fields under `company_name`. Overwrites
    any prior record for the same company (exact match on `company_name`),
    so `recall_memo()` always returns the latest assessment. No LLM call."""
    summary_text = _combined_summary(memo)
    vector = _get_embedder()([summary_text])[0]

    row = {
        "company_name": company_name,
        "combined_summary": summary_text,
        "vector": list(vector),
        "company_summary": memo.company_summary,
        "financial_summary": memo.financial_summary,
        "market_summary": memo.market_summary,
        "technical_summary": memo.technical_summary,
        "risk_summary": memo.risk_summary,
        "recommendation": memo.recommendation,
        "confidence": memo.confidence,
        "key_red_flags": list(memo.key_red_flags),
    }

    db = lancedb.connect(str(_STORAGE_PATH))
    if _TABLE_NAME in db.table_names():
        table = db.open_table(_TABLE_NAME)
        table.delete(f"company_name = '{_escape(company_name)}'")
        table.add([row])
    else:
        db.create_table(_TABLE_NAME, data=[row])


def recall_memo(company_name: str) -> dict | None:
    """Return the most recently saved structured fields for `company_name`,
    or None if no prior record exists. Exact metadata match, no LLM call."""
    db = lancedb.connect(str(_STORAGE_PATH))
    if _TABLE_NAME not in db.table_names():
        return None
    table = db.open_table(_TABLE_NAME)
    rows = (
        table.search()
        .where(f"company_name = '{_escape(company_name)}'")
        .limit(1)
        .to_list()
    )
    if not rows:
        return None
    row = rows[0]
    return {
        "company_name": row["company_name"],
        "company_summary": row["company_summary"],
        "financial_summary": row["financial_summary"],
        "market_summary": row["market_summary"],
        "technical_summary": row["technical_summary"],
        "risk_summary": row["risk_summary"],
        "recommendation": row["recommendation"],
        "confidence": row["confidence"],
        "key_red_flags": list(row["key_red_flags"]),
    }
