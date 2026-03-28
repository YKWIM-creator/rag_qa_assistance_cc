import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

logger = logging.getLogger(__name__)


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
