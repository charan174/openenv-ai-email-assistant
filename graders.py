"""
graders.py — Deterministic grading functions for each task.

Every grader takes a list of processed email dicts and returns a float
score in [0.0, 1.0].  Grading is fully reproducible and rule-based.
"""

from typing import List, Dict, Any


def grade_easy(processed: List[Dict[str, Any]], inbox: List[Dict[str, Any]]) -> float:
    """
    Grade the easy task (email classification).

    Awards 1.0 per email whose agent-assigned 'classification' matches the
    ground-truth 'label_expected'.  Returns the fraction of correct labels.

    Score range: 0.0 – 1.0
    """
    if not inbox:
        return 0.0

    total = len(inbox)
    correct = 0

    # Build a lookup of processed emails by ID
    processed_map = {p["id"]: p for p in processed}

    for email in inbox:
        eid = email["id"]
        entry = processed_map.get(eid)
        if entry and entry.get("classification") == email.get("label_expected"):
            correct += 1

    return round(correct / total, 4)


def grade_medium(processed: List[Dict[str, Any]], inbox: List[Dict[str, Any]]) -> float:
    """
    Grade the medium task (reply generation).

    For each email that requires a reply, checks:
      1. A reply was provided and has ≥ 5 words  → base credit (0.5)
      2. Reply contains at least 2 of the expected keywords → bonus (0.5)

    Returns average score across all reply-required emails.
    Score range: 0.0 – 1.0
    """
    reply_emails = [e for e in inbox if e.get("require_reply")]
    if not reply_emails:
        return 0.0

    processed_map = {p["id"]: p for p in processed}
    scores: List[float] = []

    for email in reply_emails:
        eid = email["id"]
        entry = processed_map.get(eid)
        score = 0.0

        if entry:
            reply = (entry.get("reply_text") or "").strip()
            word_count = len(reply.split())

            # Check 1: Reply exists and has sufficient length
            if word_count >= 5:
                score += 0.5

            # Check 2: Keyword relevance
            keywords = email.get("reply_keywords", [])
            if keywords:
                reply_lower = reply.lower()
                matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
                # Need at least 2 keyword matches for full bonus
                keyword_ratio = min(matched / 2.0, 1.0)
                score += 0.5 * keyword_ratio

        scores.append(score)

    return round(sum(scores) / len(scores), 4)


def grade_hard(processed: List[Dict[str, Any]], inbox: List[Dict[str, Any]]) -> float:
    """
    Grade the hard task (classification + priority + scheduling + reply).

    Per email, awards up to 1.0 broken down as:
      - Classification correct    → 0.25
      - Priority correct          → 0.25
      - Schedule valid (if req.)  → 0.25
      - Reply relevant (if req.)  → 0.25

    Components that don't apply (e.g., no schedule required) still award
    their share if the agent correctly skips them (no action needed).

    Returns average across all emails.
    Score range: 0.0 – 1.0
    """
    if not inbox:
        return 0.0

    processed_map = {p["id"]: p for p in processed}
    scores: List[float] = []

    for email in inbox:
        eid = email["id"]
        entry = processed_map.get(eid, {})
        email_score = 0.0

        # --- Classification (0.25) ---
        if entry.get("classification") == email.get("label_expected"):
            email_score += 0.25

        # --- Priority (0.25) ---
        if entry.get("priority_level") == email.get("expected_priority"):
            email_score += 0.25

        # --- Scheduling (0.25) ---
        if email.get("require_schedule"):
            schedule = entry.get("schedule_time", "")
            valid_times = email.get("valid_schedule_times", [])
            if schedule and any(vt in schedule for vt in valid_times):
                email_score += 0.25
        else:
            # No schedule required — award full credit if agent didn't schedule
            if not entry.get("schedule_time"):
                email_score += 0.25

        # --- Reply (0.25) ---
        if email.get("require_reply"):
            reply = (entry.get("reply_text") or "").strip()
            if len(reply.split()) >= 5:
                email_score += 0.125  # Half for length
                # Keyword check
                keywords = email.get("reply_keywords", [])
                if keywords:
                    reply_lower = reply.lower()
                    matched = sum(1 for kw in keywords if kw.lower() in reply_lower)
                    if matched >= 2:
                        email_score += 0.125
        else:
            # No reply required — credit if agent didn't reply
            if not entry.get("reply_text"):
                email_score += 0.25

        scores.append(email_score)

    return round(sum(scores) / len(scores), 4)


def evaluate_task(task_name: str, processed: List[Dict[str, Any]], inbox: List[Dict[str, Any]]) -> float:
    """
    Dispatch to the appropriate grader by task name.
    Returns a float score in [0.0, 1.0].
    """
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    grader = graders.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name}")
    return grader(processed, inbox)
