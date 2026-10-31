from collections import Counter


def calculate_accuracy(
    expected: list[str],
    predicted: list[str],
) -> float:
    # Reject mismatched list lengths.
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted sizes must match")

    # Prevent division by zero.
    if not expected:
        raise ValueError("Evaluation dataset cannot be empty")

    # Count all correctly classified examples.
    correct = sum(
        actual == prediction
        for actual, prediction in zip(expected, predicted, strict=True)
    )

    # Return accuracy as a value between 0 and 1.
    return correct / len(expected)


def build_confusion_matrix(
    expected: list[str],
    predicted: list[str],
) -> dict[str, dict[str, int]]:
    # Validate the input collections.
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted sizes must match")

    if not expected:
        raise ValueError("Evaluation dataset cannot be empty")

    # Identify all categories found in the dataset.
    labels = sorted(set(expected) | set(predicted))

    # Count combinations of actual and predicted labels.
    counts = Counter(zip(expected, predicted, strict=True))

    # Build rows for actual categories and columns for predictions.
    return {
        actual: {prediction: counts[(actual, prediction)] for prediction in labels}
        for actual in labels
    }
