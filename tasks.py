from typing import List, Dict, Tuple
from models import Action

EASY_INBOX = [
    {"id": "e1", "text": "Claim your $1000 gift card now! Click here.", "sender": "promo@spam.com", "priority": "unknown", "label_expected": "spam"},
    {"id": "e2", "text": "Hi, please review the attached Q3 financial report before the meeting.", "sender": "boss@company.com", "priority": "high", "label_expected": "important"},
    {"id": "e3", "text": "Are we still on for lunch tomorrow?", "sender": "friend@email.com", "priority": "normal", "label_expected": "normal"}
]

MEDIUM_INBOX = [
    {"id": "m1", "text": "Can you send me the latest metrics from the marketing campaign?", "sender": "analyst@company.com", "priority": "high", "require_reply": True},
    {"id": "m2", "text": "Just checking in, hope you are doing well.", "sender": "mom@home.com", "priority": "normal", "require_reply": True}
]

HARD_INBOX = [
    {"id": "h1", "text": "We need to schedule an urgent sync regarding the production outage. Please find a time tomorrow at 10 AM or 2 PM.", "sender": "devops@company.com", "priority": "unknown", "require_schedule": True, "require_reply": True, "require_priority": "high"}
]

TASKS_DEF = {
    "easy": {
        "inbox": EASY_INBOX,
        "description": "Classify emails as spam, important, or normal."
    },
    "medium": {
        "inbox": MEDIUM_INBOX,
        "description": "Generate appropriate replies to incoming emails."
    },
    "hard": {
        "inbox": HARD_INBOX,
        "description": "Prioritize, schedule meetings, and reply to urgent emails."
    }
}

def grade_easy(processed: List[dict]) -> float:
    if not processed:
        return 0.0
    correct = sum(1 for p in processed if p.get('classification') == p.get('label_expected'))
    return correct / len(EASY_INBOX)

def grade_medium(processed: List[dict]) -> float:
    if not processed:
        return 0.0
    scores = []
    for p in processed:
        if p.get('require_reply'):
            reply = p.get('reply_text', "")
            if reply and len(reply.split()) >= 3:
                scores.append(1.0)
            else:
                scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0

def grade_hard(processed: List[dict]) -> float:
    if not processed:
        return 0.0
    scores = []
    for p in processed:
        score = 0.0
        # Priority check
        if p.get('priority_level') == p.get('require_priority'):
            score += 0.33
        # Meeting check
        schedule = p.get('schedule_time')
        if schedule and ("10 AM" in schedule or "10:00" in schedule or "2 PM" in schedule or "14:00" in schedule):
            score += 0.33
        # Reply check
        reply = p.get('reply_text', "")
        if reply and len(reply.split()) >= 3:
            score += 0.34
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def evaluate_task(task_name: str, processed: List[dict]) -> float:
    if task_name == "easy":
        return grade_easy(processed)
    elif task_name == "medium":
        return grade_medium(processed)
    elif task_name == "hard":
        return grade_hard(processed)
    return 0.0
