# Evaluation System Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the CUNY RAG evaluation system with 5 RAGAS metrics, SQLite run history, auto-generated Markdown reports, and a semi-automated golden dataset generator with review CLI.

**Architecture:** Three modules in `src/evaluation/`: `dataset_generator.py` (LLM QA generation + review CLI), `eval.py` (rewritten with 5 metrics + SQLite persistence + terminal diff), `report.py` (Markdown report generator). Results stored in `data/eval_results.db`, reports written to `docs/eval_reports/`.

**Tech Stack:** Python 3.12, RAGAS, LangChain, SQLite (stdlib `sqlite3`), `src/generation/providers.py` for LLM access, `config/settings.py` for thresholds.

---

## Task 1: Add eval settings to config/settings.py

**Files:**
- Modify: `config/settings.py`

**Step 1: Add the new fields**

Open `config/settings.py` and add after the `api_port` field (before `cuny_senior_colleges`):

```python
    # Evaluation
    eval_db_path: str = Field(default="data/eval_results.db", env="EVAL_DB_PATH")
    eval_report_dir: str = Field(default="docs/eval_reports", env="EVAL_REPORT_DIR")
    eval_candidates_dir: str = Field(default="data/golden_dataset_candidates", env="EVAL_CANDIDATES_DIR")

    # Pass/fail thresholds
    eval_threshold_faithfulness: float = Field(default=0.80, env="EVAL_THRESHOLD_FAITHFULNESS")
    eval_threshold_answer_relevancy: float = Field(default=0.75, env="EVAL_THRESHOLD_ANSWER_RELEVANCY")
    eval_threshold_context_recall: float = Field(default=0.70, env="EVAL_THRESHOLD_CONTEXT_RECALL")
    eval_threshold_context_precision: float = Field(default=0.70, env="EVAL_THRESHOLD_CONTEXT_PRECISION")
    eval_threshold_answer_correctness: float = Field(default=0.65, env="EVAL_THRESHOLD_ANSWER_CORRECTNESS")
```

**Step 2: Verify settings load**

Run: `python -c "from config.settings import settings; print(settings.eval_db_path)"`
Expected: `data/eval_results.db`

**Step 3: Run existing settings test**

Run: `pytest tests/test_settings.py -v`
Expected: all PASS

**Step 4: Commit**

```bash
git add config/settings.py
git commit -m "feat: add eval settings (db path, report dir, thresholds)"
```

---

## Task 2: Create SQLite persistence helper in eval.py

**Files:**
- Modify: `src/evaluation/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write the failing tests**

Add to `tests/test_eval.py`:

```python
import sqlite3
import tempfile
import os
from src.evaluation.eval import init_eval_db, save_eval_run, load_last_run


def test_init_eval_db_creates_table():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        init_eval_db(db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_runs'")
        assert cursor.fetchone() is not None
        conn.close()
    finally:
        os.unlink(db_path)


def test_save_and_load_eval_run():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        init_eval_db(db_path)
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.74,
            "context_recall": 0.78,
            "context_precision": 0.80,
            "answer_correctness": 0.72,
        }
        save_eval_run(db_path, git_commit="abc1234", num_samples=10, scores=scores)
        last = load_last_run(db_path)
        assert last is not None
        assert last["faithfulness"] == 0.85
        assert last["git_commit"] == "abc1234"
        assert last["num_samples"] == 10
    finally:
        os.unlink(db_path)


def test_load_last_run_returns_none_when_empty():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        init_eval_db(db_path)
        assert load_last_run(db_path) is None
    finally:
        os.unlink(db_path)
```

**Step 2: Run to confirm they fail**

Run: `pytest tests/test_eval.py::test_init_eval_db_creates_table tests/test_eval.py::test_save_and_load_eval_run tests/test_eval.py::test_load_last_run_returns_none_when_empty -v`
Expected: FAIL with `ImportError: cannot import name 'init_eval_db'`

**Step 3: Add SQLite functions to eval.py**

Add to `src/evaluation/eval.py` (after imports):

```python
import sqlite3
import subprocess
from datetime import datetime


def init_eval_db(db_path: str) -> None:
    """Create the eval_runs table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
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
    conn.close()


def save_eval_run(db_path: str, git_commit: str, num_samples: int, scores: dict) -> None:
    """Persist one evaluation run to SQLite."""
    init_eval_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO eval_runs
           (run_at, git_commit, num_samples, faithfulness, answer_relevancy,
            context_recall, context_precision, answer_correctness)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.utcnow().isoformat(),
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
    conn.close()


def load_last_run(db_path: str) -> dict | None:
    """Return the most recent eval run as a dict, or None if no runs exist."""
    init_eval_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM eval_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"
```

**Step 4: Run tests to confirm they pass**

Run: `pytest tests/test_eval.py::test_init_eval_db_creates_table tests/test_eval.py::test_save_and_load_eval_run tests/test_eval.py::test_load_last_run_returns_none_when_empty -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/evaluation/eval.py tests/test_eval.py
git commit -m "feat: add SQLite persistence helpers to eval.py"
```

---

## Task 3: Add terminal diff output to eval.py

**Files:**
- Modify: `src/evaluation/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from io import StringIO
import sys
from src.evaluation.eval import print_eval_diff


def test_print_eval_diff_shows_new_on_first_run(capsys):
    scores = {"faithfulness": 0.85, "answer_relevancy": 0.74,
              "context_recall": 0.78, "context_precision": 0.80,
              "answer_correctness": 0.72}
    print_eval_diff(current=scores, previous=None, run_id=1, git_commit="abc1234")
    captured = capsys.readouterr()
    assert "faithfulness" in captured.out
    assert "new" in captured.out


def test_print_eval_diff_shows_delta(capsys):
    prev = {"faithfulness": 0.80, "answer_relevancy": 0.76,
            "context_recall": 0.70, "context_precision": 0.70,
            "answer_correctness": 0.65}
    curr = {"faithfulness": 0.85, "answer_relevancy": 0.74,
            "context_recall": 0.78, "context_precision": 0.80,
            "answer_correctness": 0.72}
    print_eval_diff(current=curr, previous=prev, run_id=2, git_commit="def5678")
    captured = capsys.readouterr()
    assert "+0.05" in captured.out or "+0.0500" in captured.out
    assert "↑" in captured.out
    assert "↓" in captured.out
```

**Step 2: Run to confirm they fail**

Run: `pytest tests/test_eval.py::test_print_eval_diff_shows_new_on_first_run tests/test_eval.py::test_print_eval_diff_shows_delta -v`
Expected: FAIL with `ImportError`

**Step 3: Add print_eval_diff to eval.py**

```python
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
```

**Step 4: Run tests**

Run: `pytest tests/test_eval.py::test_print_eval_diff_shows_new_on_first_run tests/test_eval.py::test_print_eval_diff_shows_delta -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/evaluation/eval.py tests/test_eval.py
git commit -m "feat: add terminal diff output to eval.py"
```

---

## Task 4: Rewrite run_evaluation() with 5 metrics + SQLite + diff

**Files:**
- Modify: `src/evaluation/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
from unittest.mock import patch, MagicMock
import tempfile, os
from src.evaluation.eval import run_evaluation


def test_run_evaluation_saves_to_db_and_returns_scores():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="CUNY is the City University of New York.", metadata={})
    ]
    mock_llm = MagicMock()

    mock_response = MagicMock()
    mock_response.answer = "CUNY stands for City University of New York."

    mock_ragas_result = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.74,
        "context_recall": 0.78,
        "context_precision": 0.80,
        "answer_correctness": 0.72,
    }

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        with patch("src.evaluation.eval.ask", return_value=mock_response), \
             patch("src.evaluation.eval.evaluate", return_value=mock_ragas_result), \
             patch("src.evaluation.eval._get_git_commit", return_value="test123"):
            result = run_evaluation(
                mock_retriever, mock_llm,
                golden_path="data/golden_dataset.json",
                db_path=db_path,
            )
        assert "faithfulness" in result
        last = load_last_run(db_path)
        assert last is not None
        assert last["git_commit"] == "test123"
    finally:
        os.unlink(db_path)
```

**Step 2: Run to confirm it fails**

Run: `pytest tests/test_eval.py::test_run_evaluation_saves_to_db_and_returns_scores -v`
Expected: FAIL (signature mismatch or missing db_path param)

**Step 3: Rewrite run_evaluation() in eval.py**

Replace the existing `run_evaluation` function with:

```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
from config.settings import settings as _settings


def run_evaluation(
    retriever,
    llm,
    golden_path: str = "data/golden_dataset.json",
    db_path: str | None = None,
) -> dict:
    """Run RAGAS evaluation on the golden dataset. Persists results to SQLite."""
    from src.generation.chain import ask

    if db_path is None:
        db_path = _settings.eval_db_path

    golden = load_golden_dataset(golden_path)
    samples = []

    for item in golden:
        question = item["question"]
        ground_truth = item["ground_truth"]
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
    ragas_result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_correctness,
        ],
    )

    scores = {
        "faithfulness": float(ragas_result["faithfulness"]),
        "answer_relevancy": float(ragas_result["answer_relevancy"]),
        "context_recall": float(ragas_result["context_recall"]),
        "context_precision": float(ragas_result["context_precision"]),
        "answer_correctness": float(ragas_result["answer_correctness"]),
    }

    previous = load_last_run(db_path)
    git_commit = _get_git_commit()

    save_eval_run(db_path, git_commit=git_commit, num_samples=len(samples), scores=scores)

    last = load_last_run(db_path)
    run_id = last["id"] if last else 1
    print_eval_diff(current=scores, previous=previous, run_id=run_id, git_commit=git_commit)

    logger.info(f"RAGAS results: {scores}")
    return scores
```

**Step 4: Run all eval tests**

Run: `pytest tests/test_eval.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/evaluation/eval.py tests/test_eval.py
git commit -m "feat: rewrite run_evaluation with 5 metrics, SQLite persistence, and diff output"
```

---

## Task 5: Create report.py

**Files:**
- Create: `src/evaluation/report.py`
- Create: `tests/test_report.py`

**Step 1: Write the failing tests**

Create `tests/test_report.py`:

```python
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
```

**Step 2: Run to confirm they fail**

Run: `pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create src/evaluation/report.py**

```python
import sqlite3
import os
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
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
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
```

**Step 4: Run tests**

Run: `pytest tests/test_report.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/evaluation/report.py tests/test_report.py
git commit -m "feat: add report.py for Markdown eval report generation"
```

---

## Task 6: Wire report generation into run_evaluation()

**Files:**
- Modify: `src/evaluation/eval.py`
- Modify: `tests/test_eval.py`

**Step 1: Write the failing test**

Add to `tests/test_eval.py`:

```python
def test_run_evaluation_generates_report():
    import tempfile, os
    from unittest.mock import patch, MagicMock
    from src.evaluation.eval import run_evaluation

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        MagicMock(page_content="CUNY is the City University of New York.", metadata={})
    ]
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.answer = "CUNY stands for City University of New York."
    mock_ragas_result = {
        "faithfulness": 0.85, "answer_relevancy": 0.74,
        "context_recall": 0.78, "context_precision": 0.80,
        "answer_correctness": 0.72,
    }

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    report_dir = tempfile.mkdtemp()

    try:
        with patch("src.evaluation.eval.ask", return_value=mock_response), \
             patch("src.evaluation.eval.evaluate", return_value=mock_ragas_result), \
             patch("src.evaluation.eval._get_git_commit", return_value="test123"):
            run_evaluation(
                mock_retriever, mock_llm,
                golden_path="data/golden_dataset.json",
                db_path=db_path,
                report_dir=report_dir,
            )
        reports = os.listdir(report_dir)
        assert len(reports) == 1
        assert reports[0].endswith("-eval-report.md")
    finally:
        os.unlink(db_path)
```

**Step 2: Run to confirm it fails**

Run: `pytest tests/test_eval.py::test_run_evaluation_generates_report -v`
Expected: FAIL (no `report_dir` param)

**Step 3: Update run_evaluation() signature and add report call**

In `src/evaluation/eval.py`, update `run_evaluation` to accept `report_dir` and call `generate_report` at the end:

```python
from src.evaluation.report import generate_report

def run_evaluation(
    retriever,
    llm,
    golden_path: str = "data/golden_dataset.json",
    db_path: str | None = None,
    report_dir: str | None = None,
) -> dict:
    # ... existing body ...
    # At the end, after print_eval_diff:
    report_path = generate_report(db_path=db_path, report_dir=report_dir)
    logger.info(f"Eval report saved to {report_path}")
    return scores
```

**Step 4: Run all eval and report tests**

Run: `pytest tests/test_eval.py tests/test_report.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/evaluation/eval.py tests/test_eval.py
git commit -m "feat: wire report generation into run_evaluation()"
```

---

## Task 7: Create dataset_generator.py

**Files:**
- Create: `src/evaluation/dataset_generator.py`
- Create: `tests/test_dataset_generator.py`

**Step 1: Write the failing tests**

Create `tests/test_dataset_generator.py`:

```python
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.evaluation.dataset_generator import generate_candidates, load_candidates


def test_generate_candidates_writes_json():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content=json.dumps([
        {"question": "What programs does Baruch offer?",
         "ground_truth": "Baruch offers business, liberal arts, and sciences."},
        {"question": "Where is Baruch located?",
         "ground_truth": "Baruch is located in Manhattan."},
    ]))

    with tempfile.TemporaryDirectory() as raw_dir, \
         tempfile.TemporaryDirectory() as candidates_dir:
        # Create a fake raw page
        school_dir = os.path.join(raw_dir, "baruch")
        os.makedirs(school_dir)
        with open(os.path.join(school_dir, "page1.md"), "w") as f:
            f.write("Baruch College offers programs in business, liberal arts, and sciences.")

        with patch("src.evaluation.dataset_generator.get_llm", return_value=mock_llm):
            generate_candidates(school="baruch", raw_dir=raw_dir, candidates_dir=candidates_dir)

        out_path = os.path.join(candidates_dir, "baruch.json")
        assert os.path.exists(out_path)
        data = json.load(open(out_path))
        assert len(data) == 2
        assert "question" in data[0]
        assert "ground_truth" in data[0]


def test_load_candidates_returns_list():
    with tempfile.TemporaryDirectory() as candidates_dir:
        data = [{"question": "Q1", "ground_truth": "A1"}]
        path = os.path.join(candidates_dir, "baruch.json")
        json.dump(data, open(path, "w"))
        result = load_candidates("baruch", candidates_dir=candidates_dir)
        assert result == data
```

**Step 2: Run to confirm they fail**

Run: `pytest tests/test_dataset_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Create src/evaluation/dataset_generator.py**

```python
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
    json.dump(all_candidates, open(out_path, "w"), indent=2, ensure_ascii=False)
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

    with open(golden_path, "w") as f:
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
```

**Step 4: Run tests**

Run: `pytest tests/test_dataset_generator.py -v`
Expected: all PASS

**Step 5: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_spider.py`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/evaluation/dataset_generator.py tests/test_dataset_generator.py
git commit -m "feat: add dataset_generator.py with LLM generation and review CLI"
```

---

## Task 8: Create docs/eval_reports/ directory and update .gitignore

**Files:**
- Create: `docs/eval_reports/.gitkeep`
- Modify: `.gitignore`

**Step 1: Create the reports directory**

```bash
mkdir -p docs/eval_reports
touch docs/eval_reports/.gitkeep
```

**Step 2: Add generated reports and SQLite db to .gitignore**

Open `.gitignore` and add:

```
# Eval artifacts
data/eval_results.db
data/golden_dataset_candidates/
docs/eval_reports/*.md
```

**Step 3: Commit**

```bash
git add docs/eval_reports/.gitkeep .gitignore
git commit -m "chore: add eval_reports dir and gitignore eval artifacts"
```

---

## Task 9: Final verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_spider.py`
Expected: all PASS, no warnings about missing imports

**Step 2: Verify settings load cleanly**

Run: `python -c "from config.settings import settings; print(settings.eval_threshold_faithfulness)"`
Expected: `0.8`

**Step 3: Verify module imports**

Run:
```bash
python -c "from src.evaluation.eval import run_evaluation, init_eval_db, print_eval_diff; print('eval ok')"
python -c "from src.evaluation.report import generate_report; print('report ok')"
python -c "from src.evaluation.dataset_generator import generate_candidates, review_candidates; print('generator ok')"
```
Expected: each prints `ok`

**Step 4: Final commit if any loose ends**

```bash
git add -p   # review carefully
git commit -m "chore: final cleanup for evaluation improvements"
```
