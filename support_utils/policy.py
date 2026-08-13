from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from support_utils.guardrails import input_guardrails


@dataclass
class Decision:
    action: str
    reason: str
    intent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def classify_intent(
    issue: str,
) -> str:
    lower_issue = issue.lower()

    if any(
        phrase in lower_issue
        for phrase in (
            "refund",
            "charged twice",
            "duplicate charge",
            "payment failed",
            "billing",
        )
    ):
        return "refund_or_billing"

    if any(
        phrase in lower_issue
        for phrase in (
            "login",
            "log in",
            "password",
            "account",
        )
    ):
        return "account_access"

    if any(
        phrase in lower_issue
        for phrase in (
            "slow",
            "bug",
            "error",
            "crash",
            "performance",
        )
    ):
        return "technical_support"

    return "general_support"


def policy_decision(
    issue: str,
    *,
    generate_candidate_before_action: bool = False,
) -> Decision:
    """
    Apply deterministic controls before model generation.

    Section 4 uses the default behavior and escalates refund
    decisions before calling the model.

    Section 6 may set generate_candidate_before_action=True so
    the model can produce a recommendation that is authorized
    separately after generation.
    """
    guardrail_result = input_guardrails(issue)

    if guardrail_result.action == "BLOCK":
        return Decision(
            action="BLOCK",
            reason=guardrail_result.reason,
            metadata=guardrail_result.metadata,
        )

    safe_input = guardrail_result.metadata.get(
        "safe_input",
        issue,
    )

    intent = classify_intent(safe_input)

    common_metadata = {
        "input_control": guardrail_result.action,
        "safe_input": safe_input,
        **guardrail_result.metadata,
    }

    if (
        intent == "refund_or_billing"
        and not generate_candidate_before_action
    ):
        return Decision(
            action="ESCALATE",
            reason=(
                "Refund and billing decisions require "
                "human verification."
            ),
            intent=intent,
            metadata={
                **common_metadata,
                "policy": (
                    "refund_requires_human_approval"
                ),
            },
        )

    return Decision(
        action=(
            "CONTINUE"
            if generate_candidate_before_action
            else "ANSWER"
        ),
        reason=(
            "Request may proceed to candidate generation."
        ),
        intent=intent,
        metadata={
            **common_metadata,
            "policy": "standard_support",
        },
    )


def authorize_recommended_action(
    next_action: str,
    *,
    customer_verified: bool,
    transaction_eligible: bool,
    amount: float,
    actor_role: str,
) -> Decision:
    """
    Apply deterministic business rules to an action recommended
    by the model.

    The model may recommend an action, but it cannot authorize it.
    """
    lower_action = next_action.lower()

    privileged_action_terms = (
        "refund",
        "reimburse",
        "reverse the charge",
        "issue a credit",
        "credit the account",
    )

    requests_privileged_action = any(
        term in lower_action
        for term in privileged_action_terms
    )

    if not requests_privileged_action:
        return Decision(
            action="EXECUTE",
            reason=(
                "No privileged financial action was requested."
            ),
            metadata={
                "policy": "standard_action",
            },
        )

    checks = {
        "customer_verified": customer_verified,
        "transaction_eligible": transaction_eligible,
        "within_automatic_limit": amount <= 50.0,
        "actor_authorized": (
            actor_role == "support_agent"
        ),
    }

    if (
        not customer_verified
        or not transaction_eligible
    ):
        return Decision(
            action="BLOCK",
            reason=(
                "Refund prerequisites were not satisfied."
            ),
            metadata={
                "checks": checks,
            },
        )

    if (
        not checks["within_automatic_limit"]
        or not checks["actor_authorized"]
    ):
        return Decision(
            action="ESCALATE",
            reason=(
                "Refund requires authorized human approval."
            ),
            metadata={
                "checks": checks,
            },
        )

    return Decision(
        action="EXECUTE",
        reason=(
            "Refund action passed deterministic authorization."
        ),
        metadata={
            "checks": checks,
        },
    )


def fallback_response(
    issue: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "status": "fallback",
        "message": (
            "I'm not able to complete that request directly, "
            "but I can route it to the right support path."
        ),
        "reason": reason,
        "next_action": "send_to_support_queue",
    }


def escalation_response(
    issue: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "status": "escalated",
        "message": (
            "This request requires human review. "
            "I'm escalating it to a support agent."
        ),
        "reason": reason,
        "next_action": "human_review",
    }


def blocked_response(
    issue: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "message": (
            "I can't process this request as written."
        ),
        "reason": reason,
        "next_action": (
            "ask_user_to_rephrase_without_sensitive_"
            "or_out_of_scope_content"
        ),
    }