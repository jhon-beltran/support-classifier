import pytest
from pydantic import ValidationError

from validation import parse_classification


def test_valid_billing_classification() -> None:
    # Define a valid JSON response.
    raw_json = '{"category": "billing"}'

    # Parse and validate the response.
    result = parse_classification(raw_json)

    # Verify the expected category.
    assert result.category == "billing"


def test_invalid_category_is_rejected() -> None:
    # Define an unsupported category.
    raw_json = '{"category": "unknown"}'

    # Verify that validation rejects it.
    with pytest.raises(ValidationError):
        parse_classification(raw_json)


def test_extra_fields_are_rejected() -> None:
    # Define a response containing an unexpected field.
    raw_json = '{"category": "billing", "debug": true}'

    # Verify that additional fields are rejected.
    with pytest.raises(ValidationError):
        parse_classification(raw_json)
