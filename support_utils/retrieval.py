from __future__ import annotations

from typing import Any, Dict, List, Set

from support_utils.data import knowledge_base
from support_utils.response_parser import tokenize


QUERY_EXPANSIONS = {
    "charged twice": {
        "billing",
        "charge",
        "duplicate",
        "duplicate_charge",
        "refund",
    },
    "duplicate charge": {
        "billing",
        "duplicate",
        "duplicate_charge",
        "refund",
    },
    "payment failed": {
        "billing",
        "charge",
        "payment",
    },
    "cannot log in": {
        "account",
        "login",
        "recovery",
    },
    "can't log in": {
        "account",
        "login",
        "recovery",
    },
    "password": {
        "account",
        "login",
        "recovery",
    },
    "slow": {
        "technical",
        "performance",
        "troubleshooting",
    },
}


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "for",
    "i",
    "in",
    "is",
    "it",
    "my",
    "need",
    "of",
    "on",
    "or",
    "still",
    "the",
    "to",
    "was",
    "what",
    "will",
    "with",
}


def prepare_search_query(query: str) -> Set[str]:
    """
    Normalize and expand the customer query.

    This is a deterministic stand-in for production query rewriting.
    """
    query_tokens = tokenize(query) - STOPWORDS
    lower_query = query.lower()

    for phrase, expansion in QUERY_EXPANSIONS.items():
        if phrase in lower_query:
            query_tokens.update(expansion)

    return query_tokens


def _document_tokens(
    document: Dict[str, Any],
) -> Set[str]:
    searchable_text = " ".join(
        [
            document["title"],
            document["text"],
            " ".join(document.get("tags", [])),
        ]
    )

    return tokenize(searchable_text)


def basic_retrieve(
    query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Naive lexical retrieval.

    This intentionally does not check source status, access, or trust.
    It can therefore return stale or ineligible evidence.
    """
    if top_k <= 0:
        return []

    query_tokens = tokenize(query)
    scored_documents = []

    for document in knowledge_base:
        document_tokens = _document_tokens(document)
        lexical_score = len(query_tokens & document_tokens)

        if lexical_score > 0:
            scored_documents.append(
                (
                    lexical_score,
                    document,
                )
            )

    scored_documents.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    return [
        document
        for _, document in scored_documents[:top_k]
    ]


def retrieve_evidence(
    query: str,
    top_k: int = 2,
    caller_role: str = "support_assistant",
    min_trust: float = 0.70,
    min_score: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Prepare, filter, and rank eligible evidence.

    The retrieval stages are:

    1. Prepare and expand the query.
    2. Filter by status, access, and trust.
    3. Require a relevant query match.
    4. Rank the remaining evidence.
    """
    if top_k <= 0:
        return []

    query_tokens = prepare_search_query(query)

    if not query_tokens:
        return []

    scored_documents = []

    for document in knowledge_base:
        if document.get("status") != "active":
            continue

        if caller_role not in document.get(
            "allowed_roles",
            [],
        ):
            continue

        if document.get("trust_score", 0) < min_trust:
            continue

        document_tokens = _document_tokens(document)
        tag_tokens = set(document.get("tags", []))

        lexical_score = len(
            query_tokens & document_tokens
        )

        tag_matches = len(
            query_tokens & tag_tokens
        )

        if lexical_score == 0 and tag_matches == 0:
            continue

        freshness_score = (
            1.0
            if document.get("freshness", "")
            >= "2026-01-01"
            else 0.25
        )

        retrieval_score = (
            lexical_score
            + (2 * tag_matches)
            + document["trust_score"]
            + freshness_score
        )

        if retrieval_score < min_score:
            continue

        scored_documents.append(
            {
                **document,
                "lexical_score": lexical_score,
                "tag_matches": tag_matches,
                "freshness_score": freshness_score,
                "retrieval_score": round(
                    retrieval_score,
                    3,
                ),
                "selection_reasons": [
                    "active_source",
                    "role_allowed",
                    "trust_threshold_met",
                    "query_match",
                ],
            }
        )

    scored_documents.sort(
        key=lambda document: (
            document["retrieval_score"],
            document["freshness"],
            document["trust_score"],
        ),
        reverse=True,
    )

    return scored_documents[:top_k]


def enhanced_retrieve(
    query: str,
    top_k: int = 2,
) -> List[Dict[str, Any]]:
    """
    Backward-compatible name used by the Section 3 notebook.
    """
    return retrieve_evidence(
        query,
        top_k=top_k,
    )


def format_retrieved_context(
    documents: List[Dict[str, Any]],
) -> str:
    if not documents:
        return "No relevant policy documents found."

    context_blocks = []

    for document in documents:
        context_blocks.append(
            f"Document: {document['title']}\n"
            f"Doc ID: {document['doc_id']}\n"
            f"Status: {document['status']}\n"
            f"Freshness: {document['freshness']}\n"
            f"Trust Score: {document['trust_score']}\n"
            f"Content: {document['text']}"
        )

    return "\n\n".join(context_blocks)


def evaluate_retrieval(
    documents: List[Dict[str, Any]],
    expected_doc_ids: Set[str],
) -> Dict[str, Any]:
    """
    Evaluate evidence selection before evaluating model generation.
    """
    retrieved_doc_ids = [
        document["doc_id"]
        for document in documents
    ]

    expected_found = (
        expected_doc_ids
        & set(retrieved_doc_ids)
    )

    coverage = (
        len(expected_found) / len(expected_doc_ids)
        if expected_doc_ids
        else 1.0
    )

    precision = (
        len(expected_found) / len(retrieved_doc_ids)
        if retrieved_doc_ids
        else 0.0
    )

    all_sources_eligible = all(
        document.get("status") == "active"
        and document.get("trust_score", 0) >= 0.70
        and "support_assistant"
        in document.get("allowed_roles", [])
        for document in documents
    )

    return {
        "retrieved_doc_ids": retrieved_doc_ids,
        "expected_doc_ids": sorted(expected_doc_ids),
        "coverage": round(coverage, 3),
        "precision_at_k": round(precision, 3),
        "top_result_expected": bool(
            retrieved_doc_ids
            and retrieved_doc_ids[0]
            in expected_doc_ids
        ),
        "all_sources_eligible": all_sources_eligible,
    }