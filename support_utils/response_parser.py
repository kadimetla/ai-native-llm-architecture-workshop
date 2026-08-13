from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


REQUIRED_FIELDS = {
    "category",
    "urgency",
    "next_action",
    "rationale",
}

VALID_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "other",
}

VALID_URGENCY = {
    "low",
    "medium",
    "high",
}


SUPPORT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": sorted(VALID_CATEGORIES),
        },
        "urgency": {
            "type": "string",
            "enum": sorted(VALID_URGENCY),
        },
        "next_action": {
            "type": "string",
            "minLength": 1,
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": sorted(REQUIRED_FIELDS),
    "additionalProperties": False,
}


RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": sorted(VALID_CATEGORIES),
        },
        "urgency": {
            "type": "string",
            "enum": sorted(VALID_URGENCY),
        },
        "next_action": {
            "type": "string",
            "minLength": 1,
        },
        "cited_doc_ids": {
            "type": "array",
            "items": {
                "type": "string",
            },
            "minItems": 1,
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
        },
    },
    "required": [
        "category",
        "urgency",
        "next_action",
        "cited_doc_ids",
        "rationale",
    ],
    "additionalProperties": False,
}


MEMORY_RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        **RAG_RESPONSE_SCHEMA["properties"],
        "memory_used": {
            "type": "boolean",
        },
    },
    "required": [
        *RAG_RESPONSE_SCHEMA["required"],
        "memory_used",
    ],
    "additionalProperties": False,
}


def tokenize(text: str) -> Set[str]:
    return set(
        re.findall(
            r"[a-zA-Z_]+",
            text.lower(),
        )
    )


def parse_json_response(
    response: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if not isinstance(response, str):
        return None, "Invalid JSON: response must be a string"

    if not response.strip():
        return None, "Invalid JSON: response is empty"

    try:
        return json.loads(response), None
    except (json.JSONDecodeError, TypeError) as error:
        return None, f"Invalid JSON: {error}"


def validate_schema_contract(
    data: Any,
    schema: Dict[str, Any],
) -> List[str]:
    """
    Validate the JSON Schema subset used by this workshop.

    This keeps the exercise dependency-free while still demonstrating
    application-side contract enforcement.
    """
    if not isinstance(data, dict):
        return ["Response must be a JSON object"]

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    errors = []

    missing_fields = required_fields - set(data)

    if missing_fields:
        errors.append(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    if schema.get("additionalProperties") is False:
        unexpected_fields = (
            set(data) - set(properties)
        )

        if unexpected_fields:
            errors.append(
                f"Unexpected fields: {sorted(unexpected_fields)}"
            )

    expected_python_types = {
        "string": str,
        "array": list,
        "boolean": bool,
        "integer": int,
        "number": (int, float),
    }

    for field_name, value in data.items():
        field_schema = properties.get(field_name)

        if field_schema is None:
            continue

        expected_type_name = field_schema.get("type")
        expected_type = expected_python_types.get(
            expected_type_name
        )

        if expected_type is not None:
            type_is_valid = isinstance(
                value,
                expected_type,
            )

            if (
                expected_type_name in {"integer", "number"}
                and isinstance(value, bool)
            ):
                type_is_valid = False

            if not type_is_valid:
                errors.append(
                    f"{field_name} must be "
                    f"{expected_type_name}"
                )
                continue

        if (
            "enum" in field_schema
            and value not in field_schema["enum"]
        ):
            errors.append(
                f"Invalid {field_name}: {value}"
            )

        if isinstance(value, str):
            minimum_length = field_schema.get(
                "minLength",
                0,
            )

            if len(value) < minimum_length:
                errors.append(
                    f"{field_name} must be a non-empty string"
                )

        if isinstance(value, list):
            minimum_items = field_schema.get(
                "minItems",
                0,
            )

            if len(value) < minimum_items:
                errors.append(
                    f"{field_name} must contain at least "
                    f"{minimum_items} item(s)"
                )

            item_schema = field_schema.get("items", {})
            item_type_name = item_schema.get("type")
            item_type = expected_python_types.get(
                item_type_name
            )

            if item_type is not None:
                for index, item in enumerate(value):
                    if not isinstance(item, item_type):
                        errors.append(
                            f"{field_name}[{index}] must be "
                            f"{item_type_name}"
                        )

    return errors


def validate_support_schema(
    data: Any,
) -> List[str]:
    return validate_schema_contract(
        data,
        SUPPORT_RESPONSE_SCHEMA,
    )