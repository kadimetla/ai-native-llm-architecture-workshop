from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Dict

from support_utils.evaluation import (
    evaluate_response,
    score_eval,
)
from support_utils.guardrails import (
    context_guardrails,
    output_guardrails,
)
from support_utils.llm_client import call_llm
from support_utils.memory import remember
from support_utils.policy import (
    authorize_recommended_action,
    blocked_response,
    escalation_response,
    fallback_response,
    policy_decision,
)
from support_utils.prompts import (
    memory_aware_rag_prompt,
)
from support_utils.response_parser import (
    MEMORY_RAG_RESPONSE_SCHEMA,
    parse_json_response,
)
from support_utils.retrieval import (
    retrieve_evidence,
)
from support_utils.tracing import (
    estimate_tokens,
    log_trace,
)


_request_counter = 0


def next_request_id() -> str:
    global _request_counter

    _request_counter += 1

    return f"req_{_request_counter:04d}"


def _finish_request(
    request_id: str,
    response: Dict[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    evaluation = evaluate_response(response)

    latency_ms = round(
        (
            time.perf_counter()
            - started_at
        )
        * 1000,
        2,
    )

    log_trace(
        request_id,
        "trace_evaluation_feedback",
        "deterministic_evaluation_completed",
        {
            "checks": evaluation,
            "score": score_eval(evaluation),
            "latency_ms": latency_ms,
        },
    )

    return response


def final_ai_native_pipeline(
    user_id: str,
    issue: str,
    prompt_version: str = (
        "v_final_evidence_contract"
    ),
    *,
    customer_verified: bool = True,
    transaction_eligible: bool = True,
    charge_amount: float = 29.0,
    actor_role: str = "support_assistant",
) -> Dict[str, Any]:
    """
    Run the eight workshop stops around a single request:

    1. Input contract
    2. Retrieve evidence
    3. Evidence selection
    4. Context assembly
    5. Model generation
    6. Output validation
    7. Decisioning and action
    8. Trace, evaluation, and feedback
    """
    request_id = next_request_id()
    started_at = time.perf_counter()

    # 1. Input contract
    decision = policy_decision(
        issue,
        generate_candidate_before_action=True,
    )

    safe_issue = (
        decision.metadata or {}
    ).get(
        "safe_input",
        issue,
    )

    log_trace(
        request_id,
        "input_contract",
        "request_checked",
        {
            "user_id": user_id,
            "prompt_version": prompt_version,
            "input_length": len(issue),
            "decision": asdict(decision),
        },
    )

    if decision.action == "BLOCK":
        response = blocked_response(
            safe_issue,
            decision.reason,
        )

        log_trace(
            request_id,
            "decisioning_and_action",
            "request_blocked",
            response,
        )

        return _finish_request(
            request_id,
            response,
            started_at,
        )

    # 2. Retrieve evidence
    retrieved_candidates = retrieve_evidence(
        safe_issue,
        top_k=5,
        caller_role=actor_role,
    )

    log_trace(
        request_id,
        "retrieve_evidence",
        "retrieval_completed",
        {
            "candidate_doc_ids": [
                document["doc_id"]
                for document
                in retrieved_candidates
            ],
            "candidate_scores": [
                document["retrieval_score"]
                for document
                in retrieved_candidates
            ],
        },
    )

    # 3. Evidence selection
    eligible_documents, excluded_documents = (
        context_guardrails(
            retrieved_candidates,
            caller_role=actor_role,
        )
    )

    selected_documents = (
        eligible_documents[:2]
    )

    log_trace(
        request_id,
        "evidence_selection",
        "eligible_evidence_selected",
        {
            "selected_doc_ids": [
                document["doc_id"]
                for document
                in selected_documents
            ],
            "excluded_documents": (
                excluded_documents
            ),
        },
    )

    if not selected_documents:
        response = fallback_response(
            safe_issue,
            (
                "No sufficient eligible evidence "
                "was found."
            ),
        )

        log_trace(
            request_id,
            "decisioning_and_action",
            "retrieval_fallback",
            response,
        )

        return _finish_request(
            request_id,
            response,
            started_at,
        )

    # 4. Context assembly
    prompt = memory_aware_rag_prompt(
        user_id,
        safe_issue,
        selected_documents,
    )

    log_trace(
        request_id,
        "context_assembly",
        "prompt_constructed",
        {
            "prompt_version": prompt_version,
            "included_doc_ids": [
                document["doc_id"]
                for document
                in selected_documents
            ],
            "estimated_prompt_tokens": (
                estimate_tokens(prompt)
            ),
        },
    )

    # 5. Model generation
    output_text = call_llm(
        prompt,
        temperature=0.2,
        response_schema=(
            MEMORY_RAG_RESPONSE_SCHEMA
        ),
        schema_name=(
            "memory_rag_support_response"
        ),
    )

    log_trace(
        request_id,
        "model_generation",
        "candidate_generated",
        {
            "estimated_output_tokens": (
                estimate_tokens(output_text)
            ),
            "output_preview": (
                output_text[:300]
            ),
        },
    )

    # 6. Output validation
    output_check = output_guardrails(
        output_text,
        MEMORY_RAG_RESPONSE_SCHEMA,
    )

    log_trace(
        request_id,
        "output_validation",
        "candidate_checked",
        asdict(output_check),
    )

    if output_check.action == "FALLBACK":
        response = fallback_response(
            safe_issue,
            output_check.reason,
        )

        log_trace(
            request_id,
            "decisioning_and_action",
            "output_fallback",
            response,
        )

        remember(
            user_id,
            safe_issue,
            json.dumps(response),
        )

        return _finish_request(
            request_id,
            response,
            started_at,
        )

    if output_check.action == "ESCALATE":
        response = escalation_response(
            safe_issue,
            output_check.reason,
        )

        log_trace(
            request_id,
            "decisioning_and_action",
            "unsafe_output_escalated",
            response,
        )

        remember(
            user_id,
            safe_issue,
            json.dumps(response),
        )

        return _finish_request(
            request_id,
            response,
            started_at,
        )

    parsed_output, parse_error = (
        parse_json_response(output_text)
    )

    if (
        parse_error
        or not isinstance(
            parsed_output,
            dict,
        )
    ):
        response = fallback_response(
            safe_issue,
            parse_error or (
                "Model output was not "
                "a JSON object."
            ),
        )

        log_trace(
            request_id,
            "decisioning_and_action",
            "parse_fallback",
            response,
        )

        return _finish_request(
            request_id,
            response,
            started_at,
        )

    # 7. Decisioning and action
    action_decision = (
        authorize_recommended_action(
            parsed_output["next_action"],
            customer_verified=(
                customer_verified
            ),
            transaction_eligible=(
                transaction_eligible
            ),
            amount=charge_amount,
            actor_role=actor_role,
        )
    )

    if action_decision.action == "BLOCK":
        response = blocked_response(
            safe_issue,
            action_decision.reason,
        )
        response["candidate"] = parsed_output

    elif (
        action_decision.action
        == "ESCALATE"
    ):
        response = escalation_response(
            safe_issue,
            action_decision.reason,
        )
        response["candidate"] = parsed_output

    else:
        response = {
            "status": "answered",
            **parsed_output,
        }

    log_trace(
        request_id,
        "decisioning_and_action",
        "action_decided",
        {
            "decision": asdict(
                action_decision
            ),
            "response_status": response[
                "status"
            ],
        },
    )

    remember(
        user_id,
        safe_issue,
        json.dumps(
            response,
            default=str,
        ),
    )

    # 8. Trace, evaluation, and feedback
    return _finish_request(
        request_id,
        response,
        started_at,
    )