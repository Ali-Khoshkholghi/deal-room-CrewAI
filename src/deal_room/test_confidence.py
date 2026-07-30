from dotenv import load_dotenv

from deal_room.crew import DealRoomCrew

load_dotenv()

# Same pitch as main.py: MRR/growth/churn/team/funding are present, but
# burn rate, cash-in-bank/runway, CAC, LTV, and gross margin are absent.
INCOMPLETE_PITCH = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.
"""

# Same shape (MRR/growth/churn/team/funding), but every field the
# incomplete pitch was missing is now explicitly filled in: burn rate,
# cash-in-bank + runway, CAC, LTV (with ratio), and gross margin.
COMPLETE_PITCH = """
Brightledger is a B2B SaaS platform that helps mid-market logistics
companies automate freight invoice reconciliation. We launched 14 months
ago and are now at $42,000 MRR, up from $9,000 MRR a year ago. Monthly
logo churn is running around 4%. We have 11 people on the team (7
engineering, 2 sales, 2 ops) and raised a $1.8M pre-seed round 18 months
ago from two angel investors and a regional fund. We're now in early
conversations for a seed round.

Our current monthly burn is $85k/month. We have $1.2M in the bank, giving
us approximately 14 months of runway at the current burn rate. Our
blended CAC is $450 per customer, and average LTV is $3,200, giving us a
7:1 LTV:CAC ratio. Gross margin is 82%.
"""


def run_once(label: str, company_info: str):
    result = DealRoomCrew().crew().kickoff(inputs={"company_info": company_info})
    assessment = result.pydantic

    print(f"\n{'=' * 70}")
    print(f"{label}")
    print("=" * 70)
    print(f"Burn rate assessment : {assessment.burn_rate_assessment}")
    print(f"Runway estimate      : {assessment.runway_estimate}")
    print(f"Unit economics notes : {assessment.unit_economics_notes}")
    print(f"Red flags            : {assessment.red_flags}")
    print(f"Data completeness    : {assessment.data_completeness}")
    print(f"Confidence           : {assessment.confidence}")

    return assessment


def run():
    complete_result = run_once("COMPLETE-DATA PITCH", COMPLETE_PITCH)
    incomplete_result_1 = run_once("INCOMPLETE PITCH - Run #1", INCOMPLETE_PITCH)
    incomplete_result_2 = run_once("INCOMPLETE PITCH - Run #2", INCOMPLETE_PITCH)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(
        f"Complete-data pitch    : data_completeness={complete_result.data_completeness}, "
        f"confidence={complete_result.confidence}"
    )
    print(
        f"Incomplete pitch run #1: data_completeness={incomplete_result_1.data_completeness}, "
        f"confidence={incomplete_result_1.confidence}"
    )
    print(
        f"Incomplete pitch run #2: data_completeness={incomplete_result_2.data_completeness}, "
        f"confidence={incomplete_result_2.confidence}"
    )


if __name__ == "__main__":
    run()
