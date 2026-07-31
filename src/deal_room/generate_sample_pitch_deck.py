"""Generates the fictional Brightledger pitch deck PDF used to test
PitchDeckReaderTool (Milestone 4, Part B/C).

Run directly (`python -m deal_room.generate_sample_pitch_deck`) to
(re)create the PDF. The content here is deliberately NOT a superset of
`main.py`'s COMPANY_INFO text -- it adds specifics (named competitors,
architecture stack, funding ask, founder background) that company_info
never mentions, so a specialist agent has an actual informational reason
to call the PDF tool rather than just re-reading what it already has.
"""

from pathlib import Path

import fitz  # PyMuPDF

SAMPLE_DECK_PATH = Path(__file__).resolve().parent / "sample_data" / "brightledger_pitch_deck.pdf"

_PAGE_SIZE = (612, 792)  # US Letter, points
_MARGIN = 72
_TITLE_SIZE = 22
_HEADING_SIZE = 15
_BODY_SIZE = 11


def _add_text_page(doc: "fitz.Document", heading: str, lines: list[str]) -> None:
    page = doc.new_page(width=_PAGE_SIZE[0], height=_PAGE_SIZE[1])
    y = _MARGIN
    page.insert_text((_MARGIN, y), heading, fontsize=_HEADING_SIZE, fontname="helv")
    y += _HEADING_SIZE + 16
    for line in lines:
        if y > _PAGE_SIZE[1] - _MARGIN:
            page = doc.new_page(width=_PAGE_SIZE[0], height=_PAGE_SIZE[1])
            y = _MARGIN
        page.insert_text((_MARGIN, y), line, fontsize=_BODY_SIZE, fontname="helv")
        y += _BODY_SIZE + 10


def build_deck(path: Path = SAMPLE_DECK_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()

    title_page = doc.new_page(width=_PAGE_SIZE[0], height=_PAGE_SIZE[1])
    title_page.insert_text((_MARGIN, 300), "Brightledger", fontsize=_TITLE_SIZE + 6, fontname="helv")
    title_page.insert_text(
        (_MARGIN, 335),
        "Seed-Stage Pitch Deck (fictional, for Milestone 4 testing)",
        fontsize=_BODY_SIZE,
        fontname="helv",
    )

    _add_text_page(
        doc,
        "Problem & Solution",
        [
            "Mid-market logistics companies reconcile freight invoices by hand",
            "against bills of lading and rate confirmations -- a process that",
            "commonly takes 3-5 finance-team hours per week per active lane.",
            "",
            "Brightledger automates this reconciliation end to end, matching",
            "carrier invoices against contracted rates and flagging discrepancies",
            "for human review instead of requiring line-by-line manual checks.",
        ],
    )

    _add_text_page(
        doc,
        "Product & Technology",
        [
            "Architecture: cloud-native on AWS, containerized services",
            "orchestrated via Kubernetes (EKS).",
            "Backend: Python / FastAPI services behind an API gateway.",
            "Data layer: PostgreSQL (system of record) + Redis (caching,",
            "rate limiting).",
            "Document ingestion: a custom OCR + rules-based parser for",
            "carrier invoice PDFs and EDI 210/214 feeds.",
            "Security: SOC 2 Type I audit currently in progress, targeting",
            "completion within two quarters; no completed certification yet.",
        ],
    )

    _add_text_page(
        doc,
        "Competitive Landscape",
        [
            "Named competitors: Tipalti (AP automation, broader scope, higher",
            "price point) and Bill.com (general AP/AR, no freight-specific",
            "matching logic).",
            "",
            "Internal analysis of our current sales pipeline shows roughly",
            "70% of prospective mid-market customers still reconcile freight",
            "invoices manually in spreadsheets today -- the primary",
            "substitute we compete against is not another vendor, but the",
            "status quo of manual Excel-based reconciliation.",
        ],
    )

    _add_text_page(
        doc,
        "Team & Funding Ask",
        [
            "Both co-founders (CEO, CTO) are first-time founders.",
            "CTO previously led a 4-person engineering team at a prior",
            "startup (pre-Series A); this is her first time scaling an",
            "engineering org past that size.",
            "",
            "Raising a $2.5M seed round at a $12M pre-money cap, primarily",
            "to fund 3 additional engineering hires and a first dedicated",
            "customer success role.",
        ],
    )

    doc.save(str(path))
    doc.close()
    return path


if __name__ == "__main__":
    result_path = build_deck()
    print(f"Sample pitch deck written to {result_path}")
