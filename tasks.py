"""
tasks.py — Task definitions for the AI Email Assistant OpenEnv environment.

Each task provides:
  - A curated inbox of emails with ground-truth annotations
  - A difficulty level (easy / medium / hard)
  - A human-readable objective for the agent
  - Expected outputs and grader criteria
"""

from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# EASY TASK — Email Classification
# Agent must label each email as spam, important, or normal.
# ---------------------------------------------------------------------------
EASY_INBOX: List[Dict[str, Any]] = [
    {
        "id": "e1",
        "subject": "🎉 You've won a $1,000 Gift Card!",
        "text": (
            "Congratulations! You have been selected as a winner of our weekly lottery. "
            "Click the link below to claim your $1,000 gift card immediately. "
            "This offer expires in 24 hours. Act NOW!"
        ),
        "sender": "promo@spam-offers.com",
        "timestamp": "2026-04-09 08:15 AM",
        "priority": "unknown",
        # Ground-truth for grading
        "label_expected": "spam",
    },
    {
        "id": "e2",
        "subject": "Q3 Financial Report — Review Required",
        "text": (
            "Hi, please review the attached Q3 financial report before our board meeting "
            "on Friday.  There are a few discrepancies in the APAC revenue numbers that "
            "need your sign-off.  Let me know if you have questions."
        ),
        "sender": "cfo@company.com",
        "timestamp": "2026-04-09 08:30 AM",
        "priority": "high",
        "label_expected": "important",
    },
    {
        "id": "e3",
        "subject": "Lunch tomorrow?",
        "text": (
            "Hey! Are we still on for lunch tomorrow at the Italian place on 5th? "
            "Let me know what time works for you."
        ),
        "sender": "alex@friends.com",
        "timestamp": "2026-04-09 08:45 AM",
        "priority": "normal",
        "label_expected": "normal",
    },
    {
        "id": "e4",
        "subject": "Your account has been compromised!",
        "text": (
            "Dear customer, we have detected suspicious activity on your account. "
            "Please verify your identity by clicking the secure link below within "
            "12 hours or your account will be permanently locked."
        ),
        "sender": "security@bank-alerts-verify.net",
        "timestamp": "2026-04-09 09:00 AM",
        "priority": "unknown",
        "label_expected": "spam",
    },
    {
        "id": "e5",
        "subject": "Project Deadline Extension",
        "text": (
            "Team, I'm extending the Project Phoenix deadline by one week to April 18. "
            "Please update your sprint plans accordingly and flag any blockers in our "
            "next standup."
        ),
        "sender": "pm@company.com",
        "timestamp": "2026-04-09 09:10 AM",
        "priority": "high",
        "label_expected": "important",
    },
]


# ---------------------------------------------------------------------------
# MEDIUM TASK — Reply Generation
# Agent must compose a contextually appropriate reply to each email.
# ---------------------------------------------------------------------------
MEDIUM_INBOX: List[Dict[str, Any]] = [
    {
        "id": "m1",
        "subject": "Latest Marketing Campaign Metrics",
        "text": (
            "Hi, can you send me the latest metrics from last month's marketing campaign? "
            "I need the click-through rates and conversion numbers for the exec summary."
        ),
        "sender": "analyst@company.com",
        "timestamp": "2026-04-09 09:15 AM",
        "priority": "high",
        "require_reply": True,
        # Keywords the reply should contain (at least some)
        "reply_keywords": ["metrics", "campaign", "send", "data", "report", "click", "conversion", "numbers"],
    },
    {
        "id": "m2",
        "subject": "Checking In",
        "text": (
            "Hey, just checking in — hope you're doing well! We should catch up soon. "
            "Let me know when you're free."
        ),
        "sender": "sara@personal.com",
        "timestamp": "2026-04-09 09:30 AM",
        "priority": "normal",
        "require_reply": True,
        "reply_keywords": ["free", "catch", "soon", "well", "thanks", "great", "time", "meet"],
    },
    {
        "id": "m3",
        "subject": "Bug Report — Login Page 500 Error",
        "text": (
            "We're seeing intermittent 500 errors on the login page after the last deploy. "
            "Can you investigate and provide an ETA on the fix?"
        ),
        "sender": "qa@company.com",
        "timestamp": "2026-04-09 09:45 AM",
        "priority": "high",
        "require_reply": True,
        "reply_keywords": ["investigate", "fix", "login", "error", "look", "issue", "deploy", "update"],
    },
]


# ---------------------------------------------------------------------------
# HARD TASK — Classification + Priority + Scheduling
# Agent must classify, set correct priority, AND schedule meetings.
# ---------------------------------------------------------------------------
HARD_INBOX: List[Dict[str, Any]] = [
    {
        "id": "h1",
        "subject": "URGENT: Production Outage Sync",
        "text": (
            "We had a critical production outage at 3 AM affecting payment processing. "
            "We need to schedule an urgent post-mortem sync tomorrow. "
            "Available slots: 10:00 AM or 2:00 PM. Please confirm which works."
        ),
        "sender": "devops-lead@company.com",
        "timestamp": "2026-04-09 07:00 AM",
        "priority": "unknown",
        "label_expected": "important",
        "expected_priority": "high",
        "require_reply": True,
        "require_schedule": True,
        # Valid schedule strings must include one of these
        "valid_schedule_times": ["10:00 AM", "10:00AM", "10 AM", "10:00", "2:00 PM", "2:00PM", "2 PM", "14:00"],
        "reply_keywords": ["outage", "sync", "confirm", "meeting", "attend", "post-mortem", "production"],
    },
    {
        "id": "h2",
        "subject": "Win a FREE iPhone 15 — Limited Offer!",
        "text": (
            "You're one of 50 lucky users chosen for a brand-new iPhone 15! "
            "Simply complete our short survey and claim your prize. "
            "No purchase necessary — hurry, offer ends tonight!"
        ),
        "sender": "deals@win-prizes-now.biz",
        "timestamp": "2026-04-09 07:30 AM",
        "priority": "unknown",
        "label_expected": "spam",
        "expected_priority": "low",
        "require_reply": False,
        "require_schedule": False,
    },
    {
        "id": "h3",
        "subject": "Quarterly Business Review Prep",
        "text": (
            "Hi, the QBR is next Wednesday at 3:00 PM. I'd like to set up a 30-minute "
            "prep call this Friday. Can you do 11:00 AM or 4:00 PM? "
            "Please also bring the customer retention dashboard."
        ),
        "sender": "vp-sales@company.com",
        "timestamp": "2026-04-09 08:00 AM",
        "priority": "unknown",
        "label_expected": "important",
        "expected_priority": "medium",
        "require_reply": True,
        "require_schedule": True,
        "valid_schedule_times": ["11:00 AM", "11:00AM", "11 AM", "11:00", "4:00 PM", "4:00PM", "4 PM", "16:00"],
        "reply_keywords": ["QBR", "prep", "Friday", "dashboard", "retention", "call", "meeting"],
    },
]


# ---------------------------------------------------------------------------
# TASK REGISTRY — central mapping consumed by environment and app
# ---------------------------------------------------------------------------
TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "inbox": EASY_INBOX,
        "description": "Classify each email as spam, important, or normal.",
        "difficulty": "easy",
        "objective": (
            "Read each email and assign the correct classification label: "
            "'spam', 'important', or 'normal'."
        ),
        "expected_output": "Each email classified with the correct label.",
    },
    "medium": {
        "inbox": MEDIUM_INBOX,
        "description": "Generate a contextually appropriate reply to each email.",
        "difficulty": "medium",
        "objective": (
            "For each email, compose a professional and relevant reply. "
            "The reply must be at least 5 words and should reference the email content."
        ),
        "expected_output": "Each email has a coherent, contextually relevant reply.",
    },
    "hard": {
        "inbox": HARD_INBOX,
        "description": (
            "Classify emails, set correct priority, schedule meetings where required, "
            "and compose relevant replies."
        ),
        "difficulty": "hard",
        "objective": (
            "For each email: (1) classify as spam/important/normal, "
            "(2) assign the correct priority level (low/medium/high), "
            "(3) schedule a meeting if required (picking one of the offered time slots), "
            "and (4) reply if required."
        ),
        "expected_output": (
            "Each email fully processed with correct classification, priority, "
            "scheduled meeting time (where applicable), and a relevant reply (where applicable)."
        ),
    },
}
