from pydantic import BaseModel


class FinancialAssessment(BaseModel):
    burn_rate_assessment: str
    runway_estimate: str
    unit_economics_notes: str
    red_flags: list[str]
    data_completeness: str  # "high" | "medium" | "low" — how much of the needed
                             # financial data (burn, runway, CAC, LTV, margin,
                             # churn) was actually present in the input
    confidence: str  # "high" | "medium" | "low" — how confident the analyst is
                      # in the company's financial health, GIVEN the available
                      # data (independent of how much data there was)
