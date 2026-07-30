from dotenv import load_dotenv

from deal_room.crew import DealRoomCrew

load_dotenv()

# Fictional test pitch: numbers are deliberately incomplete (no explicit
# burn rate, cash-in-bank, or CAC/LTV) so the agent has to flag what's
# missing rather than fabricate it.
COMPANY_INFO = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""


def run():
    result = DealRoomCrew().crew().kickoff(inputs={"company_info": COMPANY_INFO})

    assessment = result.pydantic

    print("\n=== Financial Assessment (structured) ===")
    print(f"Burn rate assessment : {assessment.burn_rate_assessment}")
    print(f"Runway estimate      : {assessment.runway_estimate}")
    print(f"Unit economics notes : {assessment.unit_economics_notes}")
    print(f"Red flags            : {assessment.red_flags}")
    print(f"Data completeness    : {assessment.data_completeness}")
    print(f"Confidence           : {assessment.confidence}")

    print("\n=== Raw pydantic object ===")
    print(repr(assessment))


if __name__ == "__main__":
    run()
