"""Semi-automated golden dataset generator.

Usage:
    # Generate candidates for a school:
    python -m src.evaluation.dataset_generator --generate --school baruch

    # Review and approve candidates interactively:
    python -m src.evaluation.dataset_generator --review --school baruch
"""
import argparse
import json
import logging
import os
from pathlib import Path

from config.settings import settings
from src.generation.providers import get_llm

logger = logging.getLogger(__name__)

GENERATE_PROMPT = """You are helping build a QA evaluation dataset.
Given the following text from a CUNY college webpage, generate {n} question-answer pairs.
Each pair must be a JSON object with keys "question" and "ground_truth".
Output ONLY a JSON array with no extra text.

Text:
{text}
"""


def generate_candidates(
    school: str,
    raw_dir: str | None = None,
    candidates_dir: str | None = None,
    n_per_page: int = 3,
) -> None:
    """Generate QA candidate pairs for a school from its raw markdown files."""
    if raw_dir is None:
        raw_dir = "data/raw"
    if candidates_dir is None:
        candidates_dir = settings.eval_candidates_dir

    os.makedirs(candidates_dir, exist_ok=True)
    school_dir = Path(raw_dir) / school
    if not school_dir.exists():
        raise FileNotFoundError(f"No raw data for school '{school}' at {school_dir}")

    llm = get_llm()
    all_candidates = []

    for page_path in sorted(school_dir.glob("*.md")):
        text = page_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        prompt = GENERATE_PROMPT.format(text=text[:2000], n=n_per_page)
        try:
            response = llm.invoke(prompt)
            pairs = json.loads(response.content)
            for pair in pairs:
                if "question" in pair and "ground_truth" in pair:
                    all_candidates.append(pair)
        except Exception as e:
            logger.warning(f"Failed to generate candidates for {page_path.name}: {e}")

    out_path = Path(candidates_dir) / f"{school}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(all_candidates)} candidates to {out_path}")


def load_candidates(school: str, candidates_dir: str | None = None) -> list[dict]:
    if candidates_dir is None:
        candidates_dir = settings.eval_candidates_dir
    path = Path(candidates_dir) / f"{school}.json"
    with open(path) as f:
        return json.load(f)


def review_candidates(
    school: str,
    golden_path: str = "data/golden_dataset.json",
    candidates_dir: str | None = None,
) -> None:
    """Interactive CLI to approve/skip/edit candidate QA pairs."""
    candidates = load_candidates(school, candidates_dir=candidates_dir)

    try:
        with open(golden_path) as f:
            golden = json.load(f)
    except FileNotFoundError:
        golden = []

    approved = 0
    for i, item in enumerate(candidates):
        print(f"\n[{i + 1}/{len(candidates)}] Question: {item['question']}")
        print(f"        Ground truth: {item['ground_truth']}")
        choice = input("        (a) approve  (s) skip  (e) edit  (q) quit\n> ").strip().lower()

        if choice == "q":
            break
        elif choice == "a":
            golden.append({"question": item["question"], "ground_truth": item["ground_truth"]})
            approved += 1
            print("✓ Added to golden_dataset.json")
        elif choice == "e":
            q = input(f"  Question [{item['question']}]: ").strip() or item["question"]
            gt = input(f"  Ground truth [{item['ground_truth']}]: ").strip() or item["ground_truth"]
            golden.append({"question": q, "ground_truth": gt})
            approved += 1
            print("✓ Edited and added to golden_dataset.json")
        # "s" → skip silently

    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {approved} pairs added. golden_dataset.json now has {len(golden)} entries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--generate", action="store_true")
    group.add_argument("--review", action="store_true")
    parser.add_argument("--school", required=True)
    parser.add_argument("--golden-path", default="data/golden_dataset.json")
    args = parser.parse_args()

    if args.generate:
        generate_candidates(school=args.school)
    else:
        review_candidates(school=args.school, golden_path=args.golden_path)
