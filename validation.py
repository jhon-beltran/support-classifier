from typing import Literal

from pydantic import BaseModel, ConfigDict

# Define the allowed customer support categories.
Category = Literal["billing", "technical_support", "general_inquiry"]


class ClassificationResult(BaseModel):
    # Reject fields not defined in this model.
    model_config = ConfigDict(extra="forbid")

    # Restrict the category to the allowed values.
    category: Category


def parse_classification(raw_json: str) -> ClassificationResult:
    # Parse JSON and validate its structure.
    return ClassificationResult.model_validate_json(raw_json)
