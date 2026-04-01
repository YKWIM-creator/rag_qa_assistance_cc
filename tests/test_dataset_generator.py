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
        with open(path, "w") as f:
            json.dump(data, f)
        result = load_candidates("baruch", candidates_dir=candidates_dir)
        assert result == data
