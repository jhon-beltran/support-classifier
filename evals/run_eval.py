import json
import os
from datetime import UTC, datetime
from pathlib import Path

from evals.metrics import build_confusion_matrix, calculate_accuracy
from main import classify_message

# Define paths relative to this script.
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset.json"
REPORT_PATH = EVAL_DIR / "latest_report.json"


def run_evaluation() -> None:
    # Load evaluation cases from the JSON dataset.
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    # Reject an empty dataset.
    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    # Prepare collections for evaluation metrics.
    expected = []
    predicted = []
    results = []

    # Run every evaluation case.
    for index, case in enumerate(dataset, start=1):
        # Call Claude using the existing classifier.
        classification = classify_message(case["input"])

        # Store the reference and generated labels.
        actual = case["expected"]
        prediction = classification.category

        expected.append(actual)
        predicted.append(prediction)

        # Compare the prediction against ground truth.
        passed = actual == prediction

        # Store the evaluation result without customer text.
        results.append(
            {
                "case_id": index,
                "expected": actual,
                "predicted": prediction,
                "passed": passed,
            }
        )

        # Display progress.
        print(f"Case {index}: {'PASS' if passed else 'FAIL'}")

    # Calculate the final evaluation matrix.
    accuracy = calculate_accuracy(expected, predicted)
    confusion_matrix = build_confusion_matrix(expected, predicted)

    # Build a machine-readable evaluation report.
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model": os.getenv("ClAUDE_MODEL", "claude-haiku-5-5"),
        "total_cases": len(dataset),
        "accuracy": accuracy,
        "confusion_matrix": confusion_matrix,
        "results": results,
    }

    # Persist the report as formatted JSON.
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Display the evaluation summary.
    print(f"\nAccuracy: {accuracy:.2%}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()
