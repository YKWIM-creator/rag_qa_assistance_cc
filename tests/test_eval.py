import pytest
import json
import sqlite3
import tempfile
import os
from src.evaluation.eval import load_golden_dataset, format_ragas_dataset
from src.evaluation.eval import init_eval_db, save_eval_run, load_last_run
from src.evaluation.eval import print_eval_diff


def test_load_golden_dataset():
    dataset = load_golden_dataset("data/golden_dataset.json")
    assert len(dataset) > 0
    assert "question" in dataset[0]
    assert "ground_truth" in dataset[0]


def test_format_ragas_dataset():
    samples = [
        {
            "question": "What is CUNY?",
            "ground_truth": "CUNY is the City University of New York.",
            "answer": "CUNY stands for City University of New York.",
            "contexts": ["CUNY is the City University of New York, a public university system."],
        }
    ]
    dataset = format_ragas_dataset(samples)
    assert dataset is not None


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
