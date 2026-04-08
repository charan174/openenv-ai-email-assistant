"""
environment.py — Core environment logic for the AI Email Assistant.

Implements the OpenEnv-compliant environment with:
  - reset()  → initialise / re-initialise the environment for a task
  - step()   → process one agent action and return (observation, reward, done, info)
  - state()  → return the full internal state as an Observation
"""

import copy
from typing import Tuple, Dict, Any

from models import Observation, Action, Reward, EnvironmentState
from tasks import TASKS
from graders import evaluate_task


class EmailAssistantEnv:
    """
    A simulated email-inbox management environment.

    The agent receives emails one at a time and must take the appropriate
    action(s) before advancing to the next email.  The environment tracks
    processed emails, computes incremental rewards, and runs a deterministic
    grader at the end of the episode.
    """

    SIMULATED_TIME = "2026-04-09 09:00 AM"

    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}"
            )
        self.task_name = task_name
        self.task_def = TASKS[task_name]
        self._state: EnvironmentState = self._build_initial_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_initial_state(self) -> EnvironmentState:
        """Create a fresh environment state for the current task."""
        inbox = copy.deepcopy(self.task_def["inbox"])
        return EnvironmentState(
            inbox=inbox,
            processed=[],
            current_index=0,
            task_name=self.task_name,
            # Allow generous steps: 4 actions per email + buffer
            max_steps=len(inbox) * 4 + 2,
            current_step=0,
            cumulative_reward=0.0,
        )

    def _current_email(self) -> Dict[str, Any] | None:
        """Return the email at the current index, or None if inbox exhausted."""
        if self._state.current_index >= len(self._state.inbox):
            return None
        return self._state.inbox[self._state.current_index]

    def _build_observation(self) -> Observation:
        """Construct an Observation from the current environment state."""
        email = self._current_email()
        if email is None:
            return Observation(
                email_id="none",
                subject="",
                email_text="All emails have been processed.",
                sender="system",
                timestamp="",
                priority="none",
                current_time=self.SIMULATED_TIME,
                unread_count=0,
                task_name=self.task_name,
                task_description=self.task_def["description"],
            )
        return Observation(
            email_id=email["id"],
            subject=email.get("subject", ""),
            email_text=email["text"],
            sender=email["sender"],
            timestamp=email.get("timestamp", ""),
            priority=email.get("priority", "unknown"),
            current_time=self.SIMULATED_TIME,
            unread_count=len(self._state.inbox) - self._state.current_index,
            task_name=self.task_name,
            task_description=self.task_def["description"],
        )

    def _should_advance(self, email: Dict[str, Any]) -> bool:
        """
        Determine whether we should advance to the next email.

        For each task difficulty, advancement happens once the primary
        objective for the current email has been fulfilled (or on skip).
        """
        if self.task_name == "easy":
            return "classification" in email
        elif self.task_name == "medium":
            return "reply_text" in email
        elif self.task_name == "hard":
            # Hard task: need classification AND priority; schedule + reply if required
            has_cls = "classification" in email
            has_pri = "priority_level" in email
            schedule_ok = (not email.get("require_schedule")) or ("schedule_time" in email)
            reply_ok = (not email.get("require_reply")) or ("reply_text" in email)
            return has_cls and has_pri and schedule_ok and reply_ok
        return False

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to initial state and return the first observation."""
        self._state = self._build_initial_state()
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one agent action.

        Returns:
            observation: the next observation
            reward: incremental Reward for this step
            done: whether the episode has ended
            info: dict with optional metadata (includes 'final_score' when done)
        """
        self._state.current_step += 1
        info: Dict[str, Any] = {}

        email = self._current_email()

        # If inbox is exhausted, episode is over
        if email is None:
            reward = Reward(value=0.0, reason="Inbox is already empty.")
            return self._build_observation(), reward, True, info

        # Validate that the action targets the current email
        if action.action_type != "skip" and action.email_id != email["id"]:
            reward = Reward(
                value=-0.2,
                reason=f"Action targets '{action.email_id}' but current email is '{email['id']}'.",
            )
            self._state.cumulative_reward += reward.value
            return self._build_observation(), reward, False, info

        # ----- Process the action -----
        reward_val = 0.0
        reason = ""

        if action.action_type == "classify_email":
            if action.classification is None:
                reward_val = -0.2
                reason = "classify_email requires a classification value."
            else:
                email["classification"] = action.classification
                # Check correctness immediately for incremental feedback
                if email.get("label_expected") and action.classification == email["label_expected"]:
                    reward_val = 0.5
                    reason = f"Correctly classified as '{action.classification}'."
                elif email.get("label_expected"):
                    reward_val = -0.2
                    reason = (
                        f"Incorrect classification '{action.classification}' "
                        f"(expected '{email['label_expected']}')."
                    )
                else:
                    reward_val = 0.5
                    reason = f"Classified as '{action.classification}'."

        elif action.action_type == "reply_email":
            if not action.reply_text or len(action.reply_text.strip()) == 0:
                reward_val = -0.2
                reason = "reply_email requires non-empty reply_text."
            else:
                email["reply_text"] = action.reply_text.strip()
                word_count = len(email["reply_text"].split())
                if word_count >= 5:
                    reward_val = 0.3
                    reason = f"Reply accepted ({word_count} words)."
                else:
                    reward_val = 0.1
                    reason = f"Reply is very short ({word_count} words). Aim for ≥ 5 words."

        elif action.action_type == "mark_priority":
            if action.priority_level is None:
                reward_val = -0.2
                reason = "mark_priority requires a priority_level value."
            else:
                email["priority_level"] = action.priority_level
                if email.get("expected_priority") and action.priority_level == email["expected_priority"]:
                    reward_val = 0.2
                    reason = f"Correctly set priority to '{action.priority_level}'."
                elif email.get("expected_priority"):
                    reward_val = -0.2
                    reason = (
                        f"Incorrect priority '{action.priority_level}' "
                        f"(expected '{email['expected_priority']}')."
                    )
                else:
                    reward_val = 0.2
                    reason = f"Priority set to '{action.priority_level}'."

        elif action.action_type == "schedule_meeting":
            if not action.schedule_time or len(action.schedule_time.strip()) == 0:
                reward_val = -0.2
                reason = "schedule_meeting requires a non-empty schedule_time."
            else:
                email["schedule_time"] = action.schedule_time.strip()
                valid_times = email.get("valid_schedule_times", [])
                if valid_times:
                    if any(vt in email["schedule_time"] for vt in valid_times):
                        reward_val = 0.2
                        reason = f"Meeting scheduled at valid time: '{email['schedule_time']}'."
                    else:
                        reward_val = -0.2
                        reason = (
                            f"Invalid schedule time '{email['schedule_time']}'. "
                            f"Valid options: {valid_times}."
                        )
                else:
                    reward_val = 0.2
                    reason = f"Meeting scheduled at '{email['schedule_time']}'."

        elif action.action_type == "skip":
            reward_val = 0.0
            reason = "Skipped the current email."

        else:
            reward_val = -0.2
            reason = f"Unknown action type '{action.action_type}'."

        # Update cumulative reward
        self._state.cumulative_reward += reward_val

        # Check if we should advance to the next email
        if self._should_advance(email) or action.action_type == "skip":
            self._state.processed.append(copy.deepcopy(email))
            self._state.current_index += 1

        # Check if episode is done
        done = (
            self._state.current_index >= len(self._state.inbox)
            or self._state.current_step >= self._state.max_steps
        )

        if done:
            final_score = evaluate_task(
                self.task_name, self._state.processed, self.task_def["inbox"]
            )
            info["final_score"] = final_score
            info["cumulative_reward"] = round(self._state.cumulative_reward, 4)
            reason += f" | Episode complete. Final grader score: {final_score:.4f}"

        reward = Reward(value=reward_val, reason=reason)
        return self._build_observation(), reward, done, info

    def get_state(self) -> Observation:
        """Return the current observation (read-only view of the environment)."""
        return self._build_observation()
