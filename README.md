---
title: AI Email Assistant Environment
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - email
  - nlp
  - agent
pinned: false
---

# 📧 AI Email Assistant Environment

An **OpenEnv-compliant** environment that simulates real-world email inbox management tasks for evaluating AI agents. Built with FastAPI, deployable on Hugging Face Spaces.

---

## Overview & Motivation

Email management is one of the most universal productivity challenges. Knowledge workers spend an average of **28% of their workday** reading and responding to emails. Automating email triage — classification, prioritization, reply drafting, and meeting scheduling — represents a high-impact application of AI agents.

This environment provides a **structured, reproducible testbed** for evaluating how well AI agents handle realistic email workflows. Unlike toy benchmarks, the tasks here mirror actual decision-making:

- **Is this email spam or important?** (requires understanding sender context and content)
- **What's an appropriate reply?** (requires natural language generation grounded in context)
- **Should I schedule a meeting? At what time?** (requires information extraction and reasoning)

---

## Architecture

```
┌──────────────┐     HTTP API      ┌────────────────────┐
│  Agent /     │ ──────────────▶   │  FastAPI Server     │
│  inference.py│ ◀──────────────   │  (app.py)           │
└──────────────┘                   │                     │
                                   │  ┌───────────────┐  │
                                   │  │ environment.py │  │
                                   │  │ (core logic)   │  │
                                   │  └───────┬───────┘  │
                                   │          │          │
                                   │  ┌───────▼───────┐  │
                                   │  │  graders.py   │  │
                                   │  │  tasks.py     │  │
                                   │  │  models.py    │  │
                                   │  └───────────────┘  │
                                   └────────────────────┘
```

---

## OpenEnv API

### `POST /reset`

Reset the environment for a given task.

**Request:**
```json
{
  "task_name": "easy"
}
```

**Response:**
```json
{
  "observation": {
    "email_id": "e1",
    "subject": "🎉 You've won a $1,000 Gift Card!",
    "email_text": "Congratulations! You have been selected...",
    "sender": "promo@spam-offers.com",
    "timestamp": "2026-04-09 08:15 AM",
    "priority": "unknown",
    "current_time": "2026-04-09 09:00 AM",
    "unread_count": 5,
    "task_name": "easy",
    "task_description": "Classify each email as spam, important, or normal."
  }
}
```

### `POST /step`

Submit an action and receive the next observation.

**Request:**
```json
{
  "action": {
    "action_type": "classify_email",
    "email_id": "e1",
    "classification": "spam"
  }
}
```

**Response:**
```json
{
  "observation": { "..." },
  "reward": 0.5,
  "done": false,
  "info": {}
}
```

### `GET /state`

Read the current observation without modifying the environment.

**Response:**
```json
{
  "observation": { "..." }
}
```

---

## Observation Space

| Field             | Type   | Description                                    |
|-------------------|--------|------------------------------------------------|
| `email_id`        | str    | Unique identifier of the current email         |
| `subject`         | str    | Subject line                                   |
| `email_text`      | str    | Full body text of the email                    |
| `sender`          | str    | Sender email address                           |
| `timestamp`       | str    | When the email was received                    |
| `priority`        | str    | Current priority label (`unknown` if unset)    |
| `current_time`    | str    | Simulated wall-clock time                      |
| `unread_count`    | int    | Number of unread emails remaining              |
| `task_name`       | str    | Active task name                               |
| `task_description`| str    | Human-readable task objective                  |

---

## Action Space

| Field            | Type                | Required For          |
|------------------|---------------------|-----------------------|
| `action_type`    | enum                | Always required       |
| `email_id`       | str                 | Always required       |
| `classification` | `spam/important/normal` | `classify_email`  |
| `reply_text`     | str                 | `reply_email`         |
| `priority_level` | `low/medium/high`   | `mark_priority`       |
| `schedule_time`  | str                 | `schedule_meeting`    |

**Action types:** `classify_email`, `reply_email`, `mark_priority`, `schedule_meeting`, `skip`

---

## Task Descriptions

### 🟢 Task 1: Email Classification (Easy)

| Property | Value |
|----------|-------|
| **Difficulty** | Easy |
| **Emails** | 5 |
| **Objective** | Classify each email as `spam`, `important`, or `normal` |
| **Grading** | Fraction of correct labels (0.0 – 1.0) |

The agent reads each email and assigns a classification label. Emails include obvious spam, important work communications, and casual personal messages.

### 🟡 Task 2: Reply Generation (Medium)

| Property | Value |
|----------|-------|
| **Difficulty** | Medium |
| **Emails** | 3 |
| **Objective** | Generate contextually appropriate replies |
| **Grading** | Reply length (≥5 words = 0.5) + keyword relevance (0.5) |

The agent must compose professional replies that demonstrate understanding of the email content. Replies are scored on both adequacy (length) and relevance (keyword matching).

### 🔴 Task 3: Full Triage (Hard)

| Property | Value |
|----------|-------|
| **Difficulty** | Hard |
| **Emails** | 3 |
| **Objective** | Classify + set priority + schedule meetings + reply |
| **Grading** | 0.25 each for classification, priority, scheduling, reply |

The agent must perform multi-step processing on each email: classify it, set the correct priority level, schedule meetings at valid times (if requested), and compose relevant replies (if needed). This tests compositional reasoning.

---

## Reward Function

The environment provides **incremental feedback** at every step:

| Action                    | Reward  | Condition                         |
|---------------------------|---------|-----------------------------------|
| Correct classification    | **+0.5**| Matches ground-truth label        |
| Wrong classification      | **-0.2**| Does not match ground-truth       |
| Good reply (≥5 words)     | **+0.3**| Non-empty, sufficient length      |
| Short reply (<5 words)    | **+0.1**| Non-empty but too brief           |
| Correct priority          | **+0.2**| Matches expected priority level   |
| Wrong priority            | **-0.2**| Does not match expected           |
| Valid meeting schedule     | **+0.2**| Time matches an offered slot      |
| Invalid meeting schedule  | **-0.2**| Time does not match any slot      |
| Skip                      | **0.0** | No penalty, no reward             |
| Mismatched email ID       | **-0.2**| Action targets wrong email        |
| Missing required fields   | **-0.2**| e.g., classify without label      |

The **final grader score** (0.0 – 1.0) is computed independently by deterministic grading functions at the end of each episode.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- pip

### Local Installation

```bash
# Clone or navigate to the project
cd ai-email-assistant

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860

# In a separate terminal, run the baseline evaluation
python inference.py
```

### Quick Run (auto-starts server)

```bash
# inference.py will auto-start the server if not running
python inference.py
```

### Using a Real LLM

```bash
export HF_TOKEN=<your_openai_or_hf_api_key>
export OPENAI_BASE_URL=https://api.openai.com/v1  # or HF inference endpoint
export LLM_MODEL=gpt-3.5-turbo
python inference.py
```

---

## Docker Instructions

### Build

```bash
docker build -t ai-email-assistant .
```

### Run (Server Mode)

```bash
docker run -p 7860:7860 ai-email-assistant
```

The API will be available at `http://localhost:7860`. Interactive docs at `http://localhost:7860/docs`.

### Run Baseline Evaluation Inside Container

```bash
docker run ai-email-assistant python inference.py
```

---

## Baseline Scores

Running `python inference.py` with the built-in deterministic mock agent produces:

```
============================================================
  AI Email Assistant — Baseline Evaluation
  Agent type: Deterministic Mock
============================================================

  TASK: EASY
  ...
  ✅ Task 'easy' complete!
     Final grader score: 1.0000

  TASK: MEDIUM
  ...
  ✅ Task 'medium' complete!
     Final grader score: 1.0000

  TASK: HARD
  ...
  ✅ Task 'hard' complete!
     Final grader score: 1.0000

============================================================
  RESULTS SUMMARY
============================================================
  Task           Score
  --------------------
  easy           1.0000
  medium         1.0000
  hard           1.0000
  --------------------
  Average        1.0000
============================================================
```

> **Note:** The mock agent is a rule-based baseline designed to achieve perfect scores for validation purposes. Real LLM agents will likely score lower depending on model capabilities.

---

## Project Structure

```
ai-email-assistant/
├── app.py              # FastAPI server (OpenEnv API endpoints)
├── models.py           # Pydantic data models (Observation, Action, Reward)
├── tasks.py            # Task definitions and email datasets
├── graders.py          # Deterministic grading functions
├── environment.py      # Core environment logic (reset, step, state)
├── inference.py        # Baseline evaluation script
├── openenv.yaml        # OpenEnv specification metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

---

## Hugging Face Deployment

This project is configured for deployment as a **Docker-based Hugging Face Space**:

1. Create a new Space on Hugging Face with SDK set to **Docker**
2. Upload all project files
3. The Space will automatically build and expose the API on port **7860**
4. Access the interactive API docs at `https://<space-url>/docs`

The `README.md` header contains the required HF Spaces metadata (title, emoji, SDK, port, tags).

---

## License

MIT
