from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from support_utils.guardrails import redact_pii


TRACE_LOGS: List[Dict[str, Any]] = []


def estimate_tokens(text: str) -> int:
    """
    Provide a dependency-free token estimate for the workshop.

    Production systems should use the tokenizer associated with
    the selected model.
    """
    if not text:
        return 0

    return max(
        1,
        len(text) // 4,
    )


def _sanitize_trace_value(
    value: Any,
) -> Any:
    if isinstance(value, str):
        return redact_pii(value)

    if isinstance(value, dict):
        return {
            key: _sanitize_trace_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_trace_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            _sanitize_trace_value(item)
            for item in value
        ]

    if isinstance(value, set):
        return sorted(
            _sanitize_trace_value(item)
            for item in value
        )

    return value


def log_trace(
    request_id: str,
    stage: str,
    event: str,
    payload: Dict[str, Any],
) -> None:
    TRACE_LOGS.append(
        {
            "timestamp": datetime.now(
                UTC
            ).isoformat(),
            "request_id": request_id,
            "stage": stage,
            "event": event,
            "payload": _sanitize_trace_value(
                payload
            ),
        }
    )


def view_traces(
    request_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if request_id is None:
        return list(TRACE_LOGS)

    return [
        trace
        for trace in TRACE_LOGS
        if trace["request_id"] == request_id
    ]


def print_traces(
    request_id: Optional[str] = None,
) -> None:
    for trace in view_traces(request_id):
        print(
            f"\n🧾 [{trace['timestamp']}]"
        )
        print(
            f"Request: {trace['request_id']}"
        )
        print(
            f"Stage: {trace['stage']} | "
            f"Event: {trace['event']}"
        )
        print(
            "Payload:",
            json.dumps(
                trace["payload"],
                indent=2,
                default=str,
            ),
        )