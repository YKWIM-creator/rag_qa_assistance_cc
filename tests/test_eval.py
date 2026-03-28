import pytest
import json
from src.evaluation.eval import load_golden_dataset, format_ragas_dataset


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
