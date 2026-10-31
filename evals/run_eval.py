import json
from pathlib import Path

from main import classify_message

# Locate the dataset relative to this script.
DATASET_PATH = Path(__file__).with_name("dataset.json")


def run_evaluation() -> None:
    # Load the labeled evaluation dataset.
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    # Track the number of correct predictions.
    correct_predictions = 0

    # Evalute each labeled message.
    for index, case in enumerate(dataset, start=1):
        # Request a classification from Clade.
        result = classify_message(case["input"])

        # Compare the prediction with the expected label.
        is_correct = result.category == case["expected"]

        # Update the score when the prediction is correct.
        if is_correct:
            correct_predictions += 1

        # Display the result for this test case.
        print(
            f"Case {index}: "
            f"expected={case['expected']}, "
            f"actual={result.category}, "
            f"passed={is_correct}"
        )

    # Calculate the overall classification accuracy.
    accuracy = correct_predictions / len(dataset)

    # Display the evaluation summary.
    print(f"\nTotal cases: {len(dataset)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    run_evaluation()
