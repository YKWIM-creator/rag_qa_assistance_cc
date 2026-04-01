import contextlib
import json
import logging
import sqlite3
import subprocess
from datetime import datetime, timezone
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

logger = logging.getLogger(__name__)


def init_eval_db(db_path: str) -> None:
    """Create the eval_runs table if it doesn't exist."""
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id                  INTEGER PRIMARY KEY,
                run_at              TEXT NOT NULL,
                git_commit          TEXT,
                num_samples         INTEGER,
                faithfulness        REAL,
                answer_relevancy    REAL,
                context_recall      REAL,
                context_precision   REAL,
                answer_correctness  REAL
            )
        """)
        conn.commit()


def save_eval_run(db_path: str, git_commit: str, num_samples: int, scores: dict) -> None:
    """Persist one evaluation run to SQLite."""
    init_eval_db(db_path)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """INSERT INTO eval_runs
               (run_at, git_commit, num_samples, faithfulness, answer_relevancy,
                context_recall, context_precision, answer_correctness)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(tz=timezone.utc).isoformat(),
                git_commit,
                num_samples,
                scores.get("faithfulness"),
                scores.get("answer_relevancy"),
                scores.get("context_recall"),
                scores.get("context_precision"),
                scores.get("answer_correctness"),
            ),
        )
        conn.commit()


def load_last_run(db_path: str) -> dict | None:
    """Return the most recent eval run as a dict, or None if no runs exist."""
    init_eval_db(db_path)
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM eval_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
    "answer_correctness",
]


def print_eval_diff(current: dict, previous: dict | None, run_id: int, git_commit: str) -> None:
    """Print a formatted table comparing current vs previous eval run."""
    col_w = 21
    print(f"\nRun #{run_id}  (commit: {git_commit})")
    print("┌" + "─" * col_w + "┬────────┬────────┬──────────┐")
    print(f"│ {'Metric':<{col_w - 2}} │  Prev  │  Now   │  Δ       │")
    print("├" + "─" * col_w + "┼────────┼────────┼──────────┤")
    for metric in METRICS:
        now = current.get(metric)
        prev_val = previous.get(metric) if previous else None
        now_str = f"{now:.2f}" if now is not None else "  —  "
        if prev_val is None:
            prev_str = "  —  "
            delta_str = "new"
        else:
            prev_str = f"{prev_val:.2f}"
            delta = now - prev_val
            arrow = "↑" if delta >= 0 else "↓"
            delta_str = f"{delta:+.4f}{arrow}"
        print(f"│ {metric:<{col_w - 2}} │ {prev_str:>6} │ {now_str:>6} │ {delta_str:<8} │")
    print("└" + "─" * col_w + "┴────────┴────────┴──────────┘")


def load_golden_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def format_ragas_dataset(samples: list[dict]) -> Dataset:
    """Convert list of QA dicts to a HuggingFace Dataset for RAGAS."""
    return Dataset.from_list(samples)


def run_evaluation(retriever, llm, golden_path: str = "data/golden_dataset.json") -> dict:
    """Run RAGAS evaluation on the golden dataset."""
    from src.generation.chain import ask

    golden = load_golden_dataset(golden_path)
    samples = []

    for item in golden:
        question = item["question"]
        ground_truth = item["ground_truth"]

        # Get RAG response
        response = ask(question, retriever, llm)
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        samples.append({
            "question": question,
            "answer": response.answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    dataset = format_ragas_dataset(samples)
    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    logger.info(f"RAGAS results: {results}")
    return results
