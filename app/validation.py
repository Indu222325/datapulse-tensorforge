"""Request validation, written by hand so the error format matches the contract exactly.

Every function returns a list of problems ({"index"?, "field", "issue"}); an empty list means valid.
Rules mirror specs/schemas/predict_request.schema.json and batch_*.schema.json.
"""
from __future__ import annotations

import re

CHANNELS = ("email", "chat", "call_transcript")
MAX_TEXT = 10_000
MAX_SUBJECT = 500
MAX_TICKET_ID = 64
_NON_SPACE = re.compile(r"\S")


def validate_ticket(obj, require_id: bool = False) -> list[dict]:
    if not isinstance(obj, dict):
        return [{"field": "ticket", "issue": "must be a JSON object"}]
    problems = []

    channel = obj.get("channel")
    if channel is None:
        problems.append({"field": "channel", "issue": "is required"})
    elif not isinstance(channel, str) or channel not in CHANNELS:
        problems.append({"field": "channel", "issue": "must be one of email, chat, call_transcript"})

    text = obj.get("text")
    if text is None:
        problems.append({"field": "text", "issue": "is required"})
    elif not isinstance(text, str):
        problems.append({"field": "text", "issue": "must be a string"})
    elif not _NON_SPACE.search(text):
        problems.append({"field": "text", "issue": "must contain at least one non-whitespace character"})
    elif len(text) > MAX_TEXT:
        problems.append({"field": "text", "issue": f"must be at most {MAX_TEXT} characters"})

    if "subject" in obj:
        subject = obj["subject"]
        if not isinstance(subject, str):
            problems.append({"field": "subject", "issue": "must be a string"})
        elif len(subject) > MAX_SUBJECT:
            problems.append({"field": "subject", "issue": f"must be at most {MAX_SUBJECT} characters"})

    if "ticket_id" in obj:
        ticket_id = obj["ticket_id"]
        if not isinstance(ticket_id, str):
            problems.append({"field": "ticket_id", "issue": "must be a string"})
        elif len(ticket_id) > MAX_TICKET_ID:
            problems.append({"field": "ticket_id", "issue": f"must be at most {MAX_TICKET_ID} characters"})
    elif require_id:
        problems.append({"field": "ticket_id", "issue": "is required"})
    return problems


def validate_batch(body, max_items: int) -> list[dict]:
    """Validates a {"tickets": [...]} body. Atomic: any problem fails the whole request."""
    if not isinstance(body, dict):
        return [{"field": "body", "issue": "must be a JSON object"}]
    tickets = body.get("tickets")
    if tickets is None:
        return [{"field": "tickets", "issue": "is required"}]
    if not isinstance(tickets, list):
        return [{"field": "tickets", "issue": "must be an array"}]
    if not 1 <= len(tickets) <= max_items:
        return [{"field": "tickets", "issue": f"must contain between 1 and {max_items} items"}]

    problems, seen = [], set()
    for index, ticket in enumerate(tickets):
        for problem in validate_ticket(ticket, require_id=True):
            problems.append({"index": index, **problem})
        if isinstance(ticket, dict) and isinstance(ticket.get("ticket_id"), str):
            if ticket["ticket_id"] in seen:
                problems.append({"index": index, "field": "ticket_id", "issue": "duplicate of an earlier item"})
            seen.add(ticket["ticket_id"])
    return problems


def clean_ticket(obj: dict) -> dict:
    """Keep only the fields the model may see. ticket_id is echoed, never used as a feature."""
    out = {"channel": obj["channel"], "subject": obj.get("subject", ""), "text": obj["text"]}
    if "ticket_id" in obj:
        out["ticket_id"] = obj["ticket_id"]
    return out
