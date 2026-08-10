"""Draft missing surface rows from canonical summaries.

The canonical row remains the source of truth. This builder only changes the
surface wording and copies the gold object from the canonical row.
"""

import argparse
import json
from pathlib import Path

if __package__:
    from .check_canonical_cases import load_canonical_cases
else:
    from check_canonical_cases import load_canonical_cases


SURFACE_PATH = Path(__file__).resolve().parents[1] / "data/surface_variants.jsonl"
TRAIN_STYLES = ("formal_english", "informal_english", "concise_english", "spelling_noise", "hinglish_roman")
VALIDATION_STYLES = ("formal_english", "informal_english")


def typo(text: str) -> str:
    for old, new in (("The", "Teh"), ("the", "teh"), (" is ", " iis "), (" was ", " wsa ")):
        if old in text:
            return text.replace(old, new, 1)
    return f"{text} Pls check."


def draft_complaint(summary: str, style: str) -> str:
    if style == "formal_english":
        return summary
    if style == "informal_english":
        return f"Please help. {summary}"
    if style == "concise_english":
        return f"Issue reported: {summary}"
    if style == "spelling_noise":
        return f"Pls help: {typo(summary)}"
    if style == "hinglish_roman":
        return f"Kripya is issue ko check karein: {summary}"
    raise ValueError(f"unknown style: {style}")


def expected_gold(canonical: dict) -> dict:
    facts = canonical["facts"]
    return {
        "service_domain": canonical["service_domain"],
        "issue_type": canonical["issue_type"],
        "location": facts["location"],
        "event_date_or_time": facts["event_date_or_time"],
        "amount_inr": facts["amount_inr"],
        "service_identifier": facts["service_identifier"],
        "urgency": canonical["urgency"],
        "missing_information": canonical["intentionally_missing_fields"],
        "formal_summary": canonical["clean_formal_summary"],
    }


def existing_pairs(path: Path = SURFACE_PATH) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (row["case_id"], row["style"])
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def build_rows(path: Path = SURFACE_PATH) -> list[dict]:
    pairs = existing_pairs(path)
    rows = []
    for canonical in load_canonical_cases():
        styles = TRAIN_STYLES if canonical["split"] == "train" else VALIDATION_STYLES
        for style in styles:
            if (canonical["case_id"], style) in pairs:
                continue
            rows.append(
                {
                    "surface_id": f"{canonical['case_id']}-{style}",
                    "case_id": canonical["case_id"],
                    "split": canonical["split"],
                    "source_type": "llm_assisted_surface",
                    "style": style,
                    "complaint": draft_complaint(canonical["clean_formal_summary"], style),
                    "gold": expected_gold(canonical),
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=145)
    args = parser.parse_args()
    for row in build_rows():
        number = int(row["case_id"].split("-")[1])
        if args.start <= number <= args.end:
            print(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
