import pytest

from evals.metrics import build_confusion_matrix, calculate_accuracy


def test_accuracy_all_correct() -> None:
    # Test a perfect classification result.
    expected = ["billing", "technical_support"]
    predicted = ["billing", "technical_support"]

    assert calculate_accuracy(expected, predicted) == 1.0


def test_accuracy_with_one_error() -> None:
    # Test accuracy with one incorrect prediction.
    expected = ["billing", "billing", "technical_support"]
    predicted = ["billing", "general_inquiry", "technical_support"]

    assert calculate_accuracy(expected, predicted) == pytest.approx(2 / 3)


def test_empty_dataset() -> None:
    # Reject an empty dataset.
    with pytest.raises(ValueError):
        calculate_accuracy([], [])


def test_mismatched_lengths() -> None:
    # Reject collections with different lengths.
    with pytest.raises(ValueError):
        calculate_accuracy(["billing"], [])


def test_confusion_matrix() -> None:
    # Define an evaluation with one error.
    expected = ["billing", "billing"]
    predicted = ["billing", "technical_support"]

    # Calculate the confusion matrix.
    matrix = build_confusion_matrix(expected, predicted)

    # Verify correct and incorrect prediction counts.
    assert matrix["billing"]["billing"] == 1
    assert matrix["billing"]["technical_support"] == 1


def test_confusion_matrix_rejects_empty_dataset() -> None:
    # Reject an empty confusion matrix dataset.
    with pytest.raises(ValueError):
        build_confusion_matrix([], [])
