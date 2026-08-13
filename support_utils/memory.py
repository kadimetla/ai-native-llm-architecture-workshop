from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List

from support_utils.guardrails import redact_pii


conversation_memory: List[Dict[str, Any]] = []

MAX_MEMORY_TEXT_LENGTH = 500


def _prepare_memory_text(text: str) -> str:
    safe_text = redact_pii(text).strip()
    return safe_text[:MAX_MEMORY_TEXT_LENGTH]


def remember(
    user_id: str,
    issue: str,
    assistant_response: str,
) -> None:
    """
    Store a small, sanitized interaction summary.

    This in-memory list is a workshop stand-in for a real memory
    store with retention, access, and deletion controls.
    """
    conversation_memory.append(
        {
            "timestamp": datetime.now(
                UTC
            ).isoformat(),
            "user_id": user_id,
            "issue": _prepare_memory_text(issue),
            "assistant_response": (
                _prepare_memory_text(
                    assistant_response
                )
            ),
        }
    )


def get_recent_memory(
    user_id: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []

    user_memories = [
        memory
        for memory in conversation_memory
        if memory["user_id"] == user_id
    ]

    return user_memories[-limit:]


def format_memory(
    memories: List[Dict[str, Any]],
) -> str:
    if not memories:
        return "No prior user history available."

    return "\n".join(
        (
            f"- Previous issue: {memory['issue']} | "
            f"Response: {memory['assistant_response']}"
        )
        for memory in memories
    )