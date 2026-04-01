import sqlite3
import tempfile
import os
from src.evaluation.report import generate_report


def _seed_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY, run_at TEXT, git_commit TEXT,
            num_samples INTEGER, faithfulness REAL, answer_relevancy REAL,
            context_recall REAL, context_precision REAL, answer_correctness REAL
        )
    """)
    rows = [
        ("2026-03-28T10:00:00", "abc1111", 20, 0.80, 0.76, 0.70, 0.68, 0.63),
        ("2026-03-29T11:00:00", "def2222", 25, 0.82, 0.77, 0.72, 0.71, 0.66),
        ("2026-03-31T14:00:00", "ghi3333", 30, 0.85, 0.74, 0.78, 0.80, 0.72),
    ]
    conn.executemany(
        "INSERT INTO eval_runs (run_at,git_commit,num_samples,faithfulness,answer_relevancy,context_recall,context_precision,answer_correctness) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_generate_report_contains_latest_scores():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    report_dir = tempfile.mkdtemp()
    try:
        _seed_db(db_path)
        report_path = generate_report(db_path=db_path, report_dir=report_dir)
        assert os.path.exists(report_path)
        content = open(report_path).read()
        assert "0.85" in content   # latest faithfulness
        assert "PASS" in content
        assert "FAIL" in content   # answer_relevancy 0.74 < 0.75 threshold
    finally:
        os.unlink(db_path)


def test_generate_report_contains_trend_table():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    report_dir = tempfile.mkdtemp()
    try:
        _seed_db(db_path)
        report_path = generate_report(db_path=db_path, report_dir=report_dir)
        content = open(report_path).read()
        assert "Trend" in content
        assert "abc1111" in content or "ghi3333" in content
    finally:
        os.unlink(db_path)
