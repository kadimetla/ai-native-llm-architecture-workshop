from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from support_utils.response_parser import (
    SUPPORT_RESPONSE_SCHEMA,
    parse_json_response,
    tokenize,
    validate_schema_contract,
)


FORBIDDEN_CLAIMS = [
    "refund processed",
    "processed your refund",
    "refund has been processed",
    "approved your refund",
    "refund has been approved",
    "issued your refund",
    "refund has been issued",
    "automatically approved",
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    action: str
    metadata: Dict[str, Any]


def contains_pii(text: str) -> bool:
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{16}\b",
    ]

    return (
        any(
            re.search(pattern, text)
            for pattern in patterns
        )
        or "ssn" in text.lower()
        or "social security number" in text.lower()
    )


def redact_pii(text: str) -> str:
    redacted = re.sub(
        r"\b\d{3}-\d{2}-\d{4}\b",
        "[REDACTED_SSN]",
        text,
    )

    redacted = re.sub(
        r"\b\d{16}\b",
        "[REDACTED_CARD_NUMBER]",
        redacted,
    )

    return redacted


def is_prompt_injection(text: str) -> bool:
    phrases = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore all instructions",
        "developer message",
        "system prompt",
        "reveal your prompt",
        "jailbreak",
        "bypass policy",
        "override policy",
    ]

    lower_text = text.lower()

    return any(
        phrase in lower_text
        for phrase in phrases
    )


def is_in_domain(text: str) -> bool:
    domain_keywords = {
        "account",
        "app",
        "billing",
        "charge",
        "charged",
        "crash",
        "error",
        "invoice",
        "login",
        "password",
        "payment",
        "refund",
        "slow",
        "subscription",
        "technical",
    }

    return bool(
        tokenize(text) & domain_keywords
    )


def input_guardrails(
    issue: str,
) -> GuardrailResult:
    if contains_pii(issue):
        return GuardrailResult(
            allowed=True,
            reason=(
                "Sensitive data was detected and redacted "
                "before downstream use."
            ),
            action="CONSTRAIN",
            metadata={
                "control": "pii_redaction",
                "safe_input": redact_pii(issue),
            },
        )

    if is_prompt_injection(issue):
        return GuardrailResult(
            allowed=False,
            reason="Prompt-injection signal detected.",
            action="BLOCK",
            metadata={
                "control": "prompt_injection",
            },
        )

    if not is_in_domain(issue):
        return GuardrailResult(
            allowed=False,
            reason=(
                "Request is outside the support "
                "assistant's permitted domain."
            ),
            action="BLOCK",
            metadata={
                "control": "domain_check",
            },
        )

    return GuardrailResult(
        allowed=True,
        reason="Input passed deterministic controls.",
        action="ALLOW",
        metadata={
            "control": "input_guardrails",
            "safe_input": issue,
        },
    )


def context_guardrails(
    documents: List[Dict[str, Any]],
    caller_role: str = "support_assistant",
    min_trust: float = 0.70,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, str]],
]:
    eligible_documents = []
    excluded_documents = []

    for document in documents:
        exclusion_reason = None

        if document.get("status") != "active":
            exclusion_reason = "inactive_source"
        elif caller_role not in document.get(
            "allowed_roles",
            [],
        ):
            exclusion_reason = "permission_denied"
        elif document.get(
            "trust_score",
            0,
        ) < min_trust:
            exclusion_reason = "below_trust_threshold"

        if exclusion_reason:
            excluded_documents.append(
                {
                    "doc_id": document["doc_id"],
                    "reason": exclusion_reason,
                }
            )
        else:
            eligible_documents.append(document)

    return (
        eligible_documents,
        excluded_documents,
    )


def output_guardrails(
    output: str,
    response_schema: Optional[Dict[str, Any]] = (
        SUPPORT_RESPONSE_SCHEMA
    ),
) -> GuardrailResult:
    lower_output = output.lower()

    if any(
        claim in lower_output
        for claim in FORBIDDEN_CLAIMS
    ):
        return GuardrailResult(
            allowed=False,
            reason=(
                "Output claims that a refund was processed "
                "or approved."
            ),
            action="ESCALATE",
            metadata={
                "control": "unsafe_refund_claim",
            },
        )

    parsed_output, parse_error = parse_json_response(
        output
    )

    if parse_error:
        return GuardrailResult(
            allowed=False,
            reason="Output is not valid JSON.",
            action="FALLBACK",
            metadata={
                "control": "json_validation",
                "error": parse_error,
            },
        )

    if not isinstance(parsed_output, dict):
        return GuardrailResult(
            allowed=False,
            reason="Output JSON is not an object.",
            action="FALLBACK",
            metadata={
                "control": "json_object_validation",
            },
        )

    if response_schema is not None:
        schema_errors = validate_schema_contract(
            parsed_output,
            response_schema,
        )

        if schema_errors:
            return GuardrailResult(
                allowed=False,
                reason=(
                    "Output does not satisfy the "
                    "response contract."
                ),
                action="FALLBACK",
                metadata={
                    "control": "schema_validation",
                    "errors": schema_errors,
                },
            )

    return GuardrailResult(
        allowed=True,
        reason="Output passed deterministic controls.",
        action="ALLOW",
        metadata={
            "control": "output_guardrails",
        },
    )