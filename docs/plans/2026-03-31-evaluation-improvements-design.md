# Evaluation System Improvements — Design Document

**Date:** 2026-03-31
**Scope:** Expand and improve the CUNY RAG evaluation pipeline

---

## 1. Overview

The current evaluation system (`src/evaluation/eval.py`) runs RAGAS with 3 metrics against a golden dataset of only 3 QA pairs. This design upgrades the evaluation system across three dimensions:

1. **Dataset scale** — Semi-automated LLM generation + human review CLI to reach ~20–30 QA pairs per school
2. **Metrics coverage** — Add `context_precision` and `answer_correctness` to the existing 3 RAGAS metrics
3. **History & reporting** — SQLite-backed run history with terminal diff output and auto-generated Markdown reports

---

## 2. Architecture

### Module Layout

```
src/evaluation/
├── __init__.py
├── eval.py              # Rewritten: run RAGAS, persist results to SQLite
├── dataset_generator.py # New: LLM QA generation + interactive review CLI
└── report.py            # New: read SQLite history, generate Markdown report + terminal diff
```

### Data Storage

```
data/
├── golden_dataset.json              # Existing (expanded via review CLI)
├── golden_dataset_candidates/       # New: LLM-generated candidates per school
│   ├── baruch.json
│   ├── hunter.json
│   └── ...
└── eval_results.db                  # New: SQLite eval run history

docs/eval_reports/
└── YYYY-MM-DD-HH-MM-eval-report.md  # Auto-generated after each run
```

---

## 3. Module Designs

### 3.1 dataset_generator.py

**Generate flow:**
1. Read markdown text from `data/raw/<school>/`
2. Call LLM to generate 3–5 QA pairs per page (question + ground_truth)
3. Write candidates to `data/golden_dataset_candidates/<school>.json`

**Review CLI:**
```
$ python -m src.evaluation.dataset_generator --review --school baruch

[1/12] Question: What GPA is required for the Baruch honors program?
        Ground truth: Students need a 3.5 GPA to qualify for the Baruch honors program.
        (a) approve  (s) skip  (e) edit  (q) quit
> a
✓ Added to golden_dataset.json
```

Approved entries are appended directly to `data/golden_dataset.json`.

---

### 3.2 eval.py (rewritten)

**Metrics (5 total):**
| Metric | Description |
|---|---|
| `faithfulness` | Is the answer grounded in the retrieved context? |
| `answer_relevancy` | Is the answer relevant to the question? |
| `context_recall` | Does retrieved context cover the ground truth? |
| `context_precision` | How much of the retrieved context is actually relevant? |
| `answer_correctness` | Semantic similarity between answer and ground truth |

**SQLite schema:**
```sql
CREATE TABLE eval_runs (
    id                  INTEGER PRIMARY KEY,
    run_at              TEXT,
    git_commit          TEXT,
    num_samples         INTEGER,
    faithfulness        REAL,
    answer_relevancy    REAL,
    context_recall      REAL,
    context_precision   REAL,
    answer_correctness  REAL
);
```

**Terminal diff output after each run:**
```
Run #5  (commit: b582fc7)
┌─────────────────────┬────────┬────────┬────────┐
│ Metric              │  Prev  │  Now   │  Δ     │
├─────────────────────┼────────┼────────┼────────┤
│ faithfulness        │  0.81  │  0.85  │ +0.04 ↑│
│ answer_relevancy    │  0.76  │  0.74  │ -0.02 ↓│
│ context_recall      │  0.70  │  0.78  │ +0.08 ↑│
│ context_precision   │   —    │  0.80  │  new   │
│ answer_correctness  │   —    │  0.72  │  new   │
└─────────────────────┴────────┴────────┴────────┘
```

---

### 3.3 report.py

Auto-generated Markdown report structure:

```markdown
# CUNY RAG Evaluation Report
**Date:** 2026-03-31 14:22  |  **Commit:** b582fc7  |  **Samples:** 47

## Latest Scores
| Metric              | Score | Threshold | Status |
|---------------------|-------|-----------|--------|
| faithfulness        | 0.85  | ≥ 0.80    | ✓ PASS |
| answer_relevancy    | 0.74  | ≥ 0.75    | ✗ FAIL |
| context_recall      | 0.78  | ≥ 0.70    | ✓ PASS |
| context_precision   | 0.80  | ≥ 0.70    | ✓ PASS |
| answer_correctness  | 0.72  | ≥ 0.65    | ✓ PASS |

## Trend (last 5 runs)
| Run | Date       | faithfulness | answer_relevancy | ... |
|-----|------------|--------------|------------------|-----|
...

## Failed Questions
- Q: "..." → answer_correctness: 0.41
```

Pass/fail thresholds are defined in `config/settings.py`.

---

## 4. Settings

New keys added to `config/settings.py`:
```python
eval_db_path: str = "data/eval_results.db"
eval_report_dir: str = "docs/eval_reports"
eval_candidates_dir: str = "data/golden_dataset_candidates"

# Pass thresholds
eval_threshold_faithfulness: float = 0.80
eval_threshold_answer_relevancy: float = 0.75
eval_threshold_context_recall: float = 0.70
eval_threshold_context_precision: float = 0.70
eval_threshold_answer_correctness: float = 0.65
```

---

## 5. Testing Strategy

- `test_dataset_generator.py` — mock LLM call, verify QA candidate format; test CLI approve/skip/edit paths
- `test_eval.py` (extended) — verify SQLite write, verify diff output format, verify all 5 metrics present
- `test_report.py` — given fixture SQLite data, verify Markdown report structure

---

## 6. Scope

**In scope:**
- `dataset_generator.py` with generate + review CLI
- `eval.py` rewrite with 5 metrics + SQLite persistence + terminal diff
- `report.py` with Markdown report generation
- New settings keys

**Out of scope:**
- LLM-as-judge metrics
- Per-school breakdown in reports (can be added later)
- CI integration / automated eval triggers
