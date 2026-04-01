import contextlib
import os
import sqlite3
from datetime import datetime

from config.settings import settings

METRICS = [
    ("faithfulness",       "eval_threshold_faithfulness"),
    ("answer_relevancy",   "eval_threshold_answer_relevancy"),
    ("context_recall",     "eval_threshold_context_recall"),
    ("context_precision",  "eval_threshold_context_precision"),
    ("answer_correctness", "eval_threshold_answer_correctness"),
]


def _load_runs(db_path: str, limit: int = 5) -> list[dict]:
    with contextlib.closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def generate_report(db_path: str | None = None, report_dir: str | None = None) -> str:
    """Generate a Markdown evaluation report from SQLite history. Returns the report path."""
    if db_path is None:
        db_path = settings.eval_db_path
    if report_dir is None:
        report_dir = settings.eval_report_dir

    os.makedirs(report_dir, exist_ok=True)
    runs = _load_runs(db_path, limit=5)
    if not runs:
        raise ValueError("No eval runs found in database.")

    latest = runs[0]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d-%H-%M")
    report_path = os.path.join(report_dir, f"{timestamp}-eval-report.md")

    lines = []
    lines.append("# CUNY RAG Evaluation Report")
    lines.append(
        f"**Date:** {latest['run_at'][:16].replace('T', ' ')}  |  "
        f"**Commit:** {latest['git_commit']}  |  "
        f"**Samples:** {latest['num_samples']}\n"
    )

    lines.append("## Latest Scores")
    lines.append("| Metric | Score | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")
    for metric, threshold_key in METRICS:
        score = latest.get(metric)
        threshold = getattr(settings, threshold_key)
        status = "✓ PASS" if score is not None and score >= threshold else "✗ FAIL"
        score_str = f"{score:.2f}" if score is not None else "—"
        lines.append(f"| {metric} | {score_str} | ≥ {threshold:.2f} | {status} |")

    lines.append("\n## Trend (last 5 runs)")
    header = "| Run | Date | " + " | ".join(m for m, _ in METRICS) + " |"
    sep = "|-----|------|" + "------|" * len(METRICS)
    lines.append(header)
    lines.append(sep)
    for run in runs:
        scores_str = " | ".join(
            f"{run.get(m):.2f}" if run.get(m) is not None else "—"
            for m, _ in METRICS
        )
        lines.append(
            f"| #{run['id']} | {run['run_at'][:10]} | {scores_str} |"
        )

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return report_path
