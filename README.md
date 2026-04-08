# AI Email Assistant Environment

This repository implements the **AI Email Assistant Environment**, complying fully with the [OpenEnv](https://github.com/openenv/openenv) specification.

## Overview
The environment simulates real-world email inbox triage, testing agents on their capability to classify, reply to, prioritize, and schedule meetings based on email content. This serves as a strong testbed for NLP capabilities and decision-making workflows.

## Environment Interface

The environment follows the standard `step`, `reset`, `state` methods mandated by OpenEnv.

### Observation Space
- `email_id` (str)
- `email_text` (str)
- `sender` (str)
- `priority` (str)
- `current_time` (str)
- `unread_count` (int)

### Action Space
- `action_type` ('classify_email', 'reply_email', 'mark_priority', 'schedule_meeting', 'skip')
- `email_id` (str)
- `classification` (optional str: 'spam', 'important', 'normal')
- `reply_text` (optional str)
- `priority_level` (optional str: 'low', 'medium', 'high')
- `schedule_time` (optional str)

### Reward Function
The reward provides dense incremental feedback:
- Correct classification -> +0.5
- Good reply -> +0.3
- Schedule meeting -> +0.2
- Marking priority -> +0.1
- Action mismatch/wrong -> -0.2

## Tasks

Provides three distinct tasks grading agents from 0.0 to 1.0:
1. **easy**: Classify incoming emails (`spam`, `important`, or `normal`).
2. **medium**: Generate appropriate replies.
3. **hard**: Triage urgent emails, prioritize, and schedule specific syncs formatting calendar times correctly.

## Deployment & Setup

### Containerized Execution (Docker)

```bash
docker build -t ai-email-assist .
docker run ai-email-assist
```

### Local Execution

```bash
pip install -r requirements.txt
python inference.py
```

### Baseline Inference Validation
Running `inference.py` exposes baseline testing simulating an OpenAI API interface (also supporting arbitrary HF_TOKEN setups).
By default, running the script with no keys executes a rule-based mock agent achieving **1.0** scores across all grader thresholds to validate deterministic behavior.
