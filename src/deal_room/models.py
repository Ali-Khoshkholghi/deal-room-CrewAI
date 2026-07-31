import difflib
from typing import Literal

from pydantic import BaseModel, field_validator


def _normalize_literal(value: object, allowed: tuple[str, ...]) -> object:
    """Coerce a near-miss string (wrong case, stray underscores/whitespace)
    to the matching entry in `allowed` before pydantic's Literal check runs.

    Manager/analyst LLM output has been observed returning
    "needs_more_diligence" instead of the declared Literal value
    "needs more diligence" — same value, underscore instead of space.
    Falls through unchanged (letting pydantic raise its normal
    ValidationError) if nothing in `allowed` is a reasonable match, so this
    never silently accepts a genuinely wrong value.
    """
    if not isinstance(value, str):
        return value
    candidate = value.strip().replace("_", " ").lower()
    for option in allowed:
        if candidate == option.lower():
            return option
    close = difflib.get_close_matches(
        candidate, [option.lower() for option in allowed], n=1, cutoff=0.6
    )
    if close:
        for option in allowed:
            if option.lower() == close[0]:
                return option
    return value


_CONFIDENCE_VALUES = ("high", "medium", "low")
_RECOMMENDATION_VALUES = ("pursue", "pass", "needs more diligence")
_MOAT_CREDIBILITY_VALUES = ("strong", "moderate", "weak", "unclear")
_RISK_LEVEL_VALUES = ("low", "medium", "high")


class FinancialAssessment(BaseModel):
    burn_rate_assessment: str
    runway_estimate: str
    unit_economics_notes: str
    red_flags: list[str]
    data_completeness: str  # "high" | "medium" | "low" — how much of the needed
                             # financial data (burn, runway, CAC, LTV, margin,
                             # churn) was actually present in the input
    confidence: Literal["high", "medium", "low"]  # how confident the analyst
                      # is in the company's financial health, GIVEN the
                      # available data (independent of how much data there was)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> object:
        return _normalize_literal(v, _CONFIDENCE_VALUES)


class MarketAssessment(BaseModel):
    market_size_notes: str
    competitive_landscape: str
    timing_assessment: str
    red_flags: list[str]


class TechnicalAssessment(BaseModel):
    moat_credibility: Literal["strong", "moderate", "weak", "unclear"]
    technical_claims_notes: str
    red_flags: list[str]

    @field_validator("moat_credibility", mode="before")
    @classmethod
    def _normalize_moat_credibility(cls, v: object) -> object:
        return _normalize_literal(v, _MOAT_CREDIBILITY_VALUES)


class RiskAssessment(BaseModel):
    regulatory_risks: list[str]
    team_risks: list[str]
    market_timing_risks: list[str]
    overall_risk_level: Literal["low", "medium", "high"]

    @field_validator("overall_risk_level", mode="before")
    @classmethod
    def _normalize_overall_risk_level(cls, v: object) -> object:
        return _normalize_literal(v, _RISK_LEVEL_VALUES)


class InvestmentMemo(BaseModel):
    company_summary: str
    financial_summary: str
    market_summary: str
    technical_summary: str
    risk_summary: str
    key_red_flags: list[str]
    recommendation: Literal["pursue", "pass", "needs more diligence"]
    confidence: Literal["high", "medium", "low"]

    @field_validator("recommendation", mode="before")
    @classmethod
    def _normalize_recommendation(cls, v: object) -> object:
        return _normalize_literal(v, _RECOMMENDATION_VALUES)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_memo_confidence(cls, v: object) -> object:
        return _normalize_literal(v, _CONFIDENCE_VALUES)
