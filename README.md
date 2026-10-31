# CCDV-F Lab 01 — Claude Support Classifier

A Python CLI application that classifies customer support messages using the Anthropic Claude API.

## Tech Stack

- Python 3.14
- uv
- Anthropic Python SDK
- Pydantic
- pytest
- Ruff

## Features

- Customer support classification
- JSON Schema structured outputs
- Response validation
- Token usage reporting
- Automated validation tests

## Setup

Install uv and synchronize dependencies:

```bash
uv sync --locked
```

Configure the API key in your terminal:

```bash
export ANTHROPIC_API_KEY="YOUR_API_KEY"
```

## Usage

```bash
uv run python main.py "I was charged twice this month."
```

Expected classification:

```json
{"category": "billing"}
```

## Quality Checks

```bash
uv run pytest -v
uv run ruff check .
uv run ruff format --check .
```

## Learning Objectives

- Integrate Claude using the official Python SDK.
- Generate structured JSON responses.
- Validate model output with Pydantic.
- Manage reproducible dependencies with uv.
- Apply testing and code-quality practices.
- Version and publish code using Git and GitHub.

## Security

API keys must be provided through environment variables. Never commit secrets, customer data, or local environment files.

## Certification

Claude Certified Developer – Foundations (CCDV-F)

Lab 01: API Integration and Structured Output Validation.
