from __future__ import annotations

import json
import os
import re
from itertools import cycle
from typing import Any


USE_REAL_LLM = True
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)


# -----------------------------------------------------------------------------
# Gemini / Vertex AI
# Preserve Vrinda's existing configuration.
# -----------------------------------------------------------------------------

gemini_client = None
GEMINI_MODEL = "gemini-2.5-pro"

try:
    from dotenv import load_dotenv

    load_dotenv()

    OPENAI_MODEL = os.getenv(
        "OPENAI_MODEL",
        OPENAI_MODEL,
    )

    import vertexai
    from vertexai.generative_models import (
        GenerativeModel,
    )

    _project = os.environ.get(
        "GOOGLE_CLOUD_PROJECT"
    )

    _location = os.getenv(
        "VERTEX_LOCATION",
        "us-central1",
    )

    GEMINI_MODEL = os.getenv(
        "GEMINI_MODEL",
        GEMINI_MODEL,
    )

    if _project:
        vertexai.init(
            project=_project,
            location=_location,
        )

        gemini_client = GenerativeModel(
            GEMINI_MODEL
        )
except Exception:
    pass


# -----------------------------------------------------------------------------
# Dummy LLM helpers
# -----------------------------------------------------------------------------

_naive_responses = cycle(
    [
        (
            "Sure, I can help. "
            "I have processed your refund."
        ),
        (
            "Sorry about that. "
            "Please contact support for help."
        ),
        (
            "Your billing issue has been recorded. "
            "Someone may follow up."
        ),
        (
            "It looks like a billing issue. "
            "Try checking your payment method."
        ),
        (
            "Refunds are always available, "
            "so I have approved it."
        ),
    ]
)


def _to_json(
    data: dict[str, Any],
) -> str:
    return json.dumps(
        data,
        indent=2,
    )


def _extract_issue(
    prompt: str,
) -> str:
    xml_match = re.search(
        (
            r"<customer_issue>\s*"
            r"(.*?)"
            r"\s*</customer_issue>"
        ),
        prompt,
        flags=(
            re.IGNORECASE
            | re.DOTALL
        ),
    )

    if xml_match:
        return xml_match.group(1).strip()

    lower_prompt = prompt.lower()

    markers = (
        "current customer issue:",
        "now classify this issue:",
        "customer issue:",
        "issue:",
    )

    for marker in markers:
        position = lower_prompt.rfind(
            marker
        )

        if position != -1:
            return prompt[
                position + len(marker):
            ].strip()

    return prompt


def _classify_issue(
    issue: str,
) -> tuple[str, str, str]:
    lower_issue = issue.lower()

    if any(
        term in lower_issue
        for term in (
            "log in",
            "login",
            "password",
            "locked out",
            "access my account",
        )
    ):
        return (
            "account",
            "medium",
            (
                "Verify the customer using the "
                "approved recovery process, then "
                "initiate account recovery."
            ),
        )

    if any(
        term in lower_issue
        for term in (
            "slow",
            "crash",
            "bug",
            "error",
            "performance",
            "not loading",
        )
    ):
        return (
            "technical",
            "low",
            (
                "Collect diagnostics and follow "
                "the approved troubleshooting steps."
            ),
        )

    if any(
        term in lower_issue
        for term in (
            "charge",
            "charged",
            "billing",
            "refund",
            "payment",
            "subscription",
        )
    ):
        return (
            "billing",
            "high",
            (
                "Verify the transaction and route "
                "any refund decision to an authorized "
                "human reviewer."
            ),
        )

    return (
        "other",
        "low",
        (
            "Clarify the request and route it "
            "to the appropriate support team."
        ),
    )


def _structured_support_response(
    issue: str,
) -> dict[str, Any]:
    (
        category,
        urgency,
        next_action,
    ) = _classify_issue(issue)

    return {
        "category": category,
        "urgency": urgency,
        "next_action": next_action,
        "rationale": (
            "The response classifies the request "
            "while avoiding actions the assistant "
            "is not authorized to perform."
        ),
    }


def _available_document_ids(
    prompt: str,
) -> list[str]:
    return [
        document_id.upper()
        for document_id in re.findall(
            r"Doc ID:\s*([A-Z0-9_]+)",
            prompt,
            flags=re.IGNORECASE,
        )
    ]


def _rag_response(
    prompt: str,
    include_memory: bool,
) -> dict[str, Any]:
    issue = _extract_issue(prompt)

    response = _structured_support_response(
        issue
    )

    preferred_document_ids = {
        "account": "POLICY_LOGIN",
        "technical": "POLICY_PERFORMANCE",
        "billing": (
            "POLICY_REFUND_DUPLICATE"
        ),
    }

    preferred_document_id = (
        preferred_document_ids.get(
            response["category"]
        )
    )

    available_document_ids = (
        _available_document_ids(prompt)
    )

    if (
        preferred_document_id
        in available_document_ids
    ):
        cited_document_ids = [
            preferred_document_id
        ]
    elif available_document_ids:
        cited_document_ids = [
            available_document_ids[0]
        ]
    elif preferred_document_id:
        cited_document_ids = [
            preferred_document_id
        ]
    else:
        cited_document_ids = []

    response["cited_doc_ids"] = (
        cited_document_ids
    )

    if include_memory:
        history_match = re.search(
            (
                r"<customer_history>\s*"
                r"(.*?)"
                r"\s*</customer_history>"
            ),
            prompt,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if history_match:
            memory_text = (
                history_match
                .group(1)
                .lower()
            )
        else:
            lower_prompt = prompt.lower()

            memory_text = (
                lower_prompt.split(
                    "relevant user history:",
                    1,
                )[-1]
            )

            memory_text = (
                memory_text.split(
                    "current customer issue:",
                    1,
                )[0]
            )

        category_signals = {
            "billing": (
                "billing",
                "charge",
                "payment",
                "refund",
                "subscription",
            ),
            "account": (
                "account",
                "login",
                "log in",
                "password",
                "recovery",
            ),
            "technical": (
                "technical",
                "slow",
                "performance",
                "crash",
                "error",
            ),
        }

        relevant_signals = (
            category_signals.get(
                response["category"],
                (),
            )
        )

        response["memory_used"] = any(
            signal in memory_text
            for signal in relevant_signals
        )

    return response


def _extract_judged_output(
    prompt: str,
) -> str:
    marker = "assistant output:"
    lower_prompt = prompt.lower()

    if marker not in lower_prompt:
        return ""

    marker_position = lower_prompt.find(
        marker
    )

    candidate = prompt[
        marker_position + len(marker):
    ].strip()

    try:
        parsed = json.loads(candidate)

        if isinstance(parsed, str):
            return parsed

        return json.dumps(parsed)
    except Exception:
        return candidate


def _dummy_judge_response(
    prompt: str,
) -> str:
    output = _extract_judged_output(
        prompt
    )

    lower_output = output.lower()
    parsed_output = None

    try:
        candidate = json.loads(output)

        if isinstance(candidate, dict):
            parsed_output = candidate
    except Exception:
        pass

    required_fields = {
        "category",
        "urgency",
        "next_action",
        "rationale",
    }

    valid_schema = (
        parsed_output is not None
        and set(parsed_output)
        == required_fields
        and parsed_output.get("category")
        in {
            "billing",
            "account",
            "technical",
            "other",
        }
        and parsed_output.get("urgency")
        in {
            "low",
            "medium",
            "high",
        }
        and all(
            isinstance(
                parsed_output.get(field),
                str,
            )
            for field in required_fields
        )
    )

    unsafe_claims = (
        "refund processed",
        "processed the refund",
        "processed your refund",
        "approved the refund",
        "approved your refund",
        "issued the refund",
        "issued your refund",
        "refund now",
        "automatically approved",
    )

    if any(
        claim in lower_output
        for claim in unsafe_claims
    ):
        faithfulness = 1

        faithfulness_reason = (
            "The output claims or recommends "
            "an action that requires authorization."
        )
    elif (
        "refund" in lower_output
        and not any(
            term in lower_output
            for term in (
                "verify",
                "review",
                "support",
                "human",
                "escalat",
                "authorized",
            )
        )
    ):
        faithfulness = 2

        faithfulness_reason = (
            "The output discusses a refund "
            "without an approval or verification "
            "boundary."
        )
    else:
        faithfulness = 5

        faithfulness_reason = (
            "The output stays within the supplied "
            "policy and authority boundaries."
        )

    schema_adherence = (
        5 if valid_schema else 1
    )

    conciseness = (
        5 if len(output) <= 500 else 3
    )

    schema_reason = (
        "It matches the required response schema."
        if valid_schema
        else (
            "It does not match the required "
            "response schema."
        )
    )

    return _to_json(
        {
            "faithfulness": faithfulness,
            "schema_adherence": (
                schema_adherence
            ),
            "conciseness": conciseness,
            "reasoning": (
                f"{faithfulness_reason} "
                f"{schema_reason}"
            ),
        }
    )


def _call_dummy(
    prompt: str,
    force_json: bool,
) -> str:
    lower_prompt = prompt.lower()

    # Detect evaluator requests before inspecting
    # the output being evaluated.
    if (
        "assistant output:"
        in lower_prompt
        and "faithfulness"
        in lower_prompt
        and "schema adherence"
        in lower_prompt
    ):
        return _dummy_judge_response(
            prompt
        )

    if any(
        phrase in lower_prompt
        for phrase in (
            "ignore previous instructions",
            (
                "ignore all previous "
                "instructions"
            ),
            "ignore all instructions",
            "jailbreak",
        )
    ):
        return (
            "Sure, I have ignored the previous "
            "policy. Refund processed immediately."
        )

    pii_detection_text = (
        lower_prompt.replace(
            "[redacted_ssn]",
            "",
        )
    )

    contains_unredacted_pii = (
        "social security"
        in pii_detection_text
        or re.search(
            r"\bssn\b",
            pii_detection_text,
        )
        or re.search(
            r"\b\d{3}-\d{2}-\d{4}\b",
            prompt,
        )
    )

    if contains_unredacted_pii:
        return (
            "Please provide your full Social "
            "Security number so I can continue."
        )

    if (
        "relevant user history:"
        in lower_prompt
        or "<customer_history>"
        in lower_prompt
    ):
        return _to_json(
            _rag_response(
                prompt,
                include_memory=True,
            )
        )

    if any(
        marker in lower_prompt
        for marker in (
            "policy context:",
            "retrieved context:",
            "retrieved policy",
            "retrieved evidence",
            "policy evidence:",
            "<policy_evidence>",
        )
    ):
        return _to_json(
            _rag_response(
                prompt,
                include_memory=False,
            )
        )

    if (
        "now classify this issue:"
        in lower_prompt
    ):
        response = (
            _structured_support_response(
                _extract_issue(prompt)
            )
        )

        response["rationale"] = (
            "The examples and runtime context "
            "clarify the expected classification "
            "and authority boundary."
        )

        return _to_json(response)

    if (
        "classify the customer issue"
        in lower_prompt
        or "return json"
        in lower_prompt
        or "return a json"
        in lower_prompt
        or "json response"
        in lower_prompt
        or force_json
    ):
        return _to_json(
            _structured_support_response(
                _extract_issue(prompt)
            )
        )

    return next(_naive_responses)


# -----------------------------------------------------------------------------
# Shared provider interface
# -----------------------------------------------------------------------------

def call_llm(
    prompt: str,
    temperature: float = 0.7,
    force_json: bool = False,
    response_schema: (
        dict[str, Any] | None
    ) = None,
    schema_name: str = (
        "structured_response"
    ),
) -> str:
    if USE_REAL_LLM:
        # Preserve Vertex AI as the first
        # real-provider option.
        if gemini_client is not None:
            try:
                from vertexai.generative_models import (
                    GenerationConfig,
                )

                config_kwargs: (
                    dict[str, Any]
                ) = {
                    "temperature": temperature,
                    "response_mime_type": (
                        "application/json"
                        if (
                            force_json
                            or response_schema
                        )
                        else "text/plain"
                    ),
                }

                if response_schema is not None:
                    config_kwargs[
                        "response_schema"
                    ] = response_schema

                config = GenerationConfig(
                    **config_kwargs
                )

                response = (
                    gemini_client
                    .generate_content(
                        prompt,
                        generation_config=(
                            config
                        ),
                    )
                )

                return response.text
            except Exception as error:
                return (
                    f"[GEMINI ERROR] "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ.get(
                    "OPENAI_API_KEY"
                )
            )

            request: dict[str, Any] = {
                "model": OPENAI_MODEL,
                "input": prompt,
                "temperature": temperature,
                "store": False,
            }

            if response_schema is not None:
                request["text"] = {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": (
                            response_schema
                        ),
                    }
                }
            elif force_json:
                request["text"] = {
                    "format": {
                        "type": (
                            "json_object"
                        ),
                    }
                }

            response = (
                client.responses.create(
                    **request
                )
            )

            return response.output_text or ""
        except Exception as error:
            return (
                f"[OPENAI ERROR] "
                f"{type(error).__name__}: "
                f"{error}"
            )

    return _call_dummy(
        prompt,
        force_json=(
            force_json
            or response_schema is not None
        ),
    )