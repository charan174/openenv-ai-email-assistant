from typing import Tuple, Dict, Any
from models import Observation, Action, Reward, State
from tasks import TASKS_DEF, evaluate_task
import copy

class EmailAssistantEnv:
    def __init__(self, task_name: str = "easy"):
        if task_name not in TASKS_DEF:
            raise ValueError(f"Unknown task: {task_name}")
        self.task_name = task_name
        self.task_def = TASKS_DEF[task_name]
        self._state = State(
            inbox=copy.deepcopy(self.task_def["inbox"]),
            processed=[],
            current_index=0,
            task_name=self.task_name,
            max_steps=len(self.task_def["inbox"]) * 3, # Allow multiple actions per email
            current_steps=0
        )
        self.current_time = "2026-04-09 09:00 AM"

    def _get_obs(self) -> Observation:
        if self._state.current_index >= len(self._state.inbox):
            # Empty observation indicating done
            return Observation(
                email_id="none",
                email_text="Inbox is empty.",
                sender="system",
                priority="none",
                current_time=self.current_time,
                unread_count=0
            )
        
        email = self._state.inbox[self._state.current_index]
        return Observation(
            email_id=email["id"],
            email_text=email["text"],
            sender=email["sender"],
            priority=email.get("priority", "none"),
            current_time=self.current_time,
            unread_count=len(self._state.inbox) - self._state.current_index
        )

    def reset(self) -> Observation:
        self._state = State(
            inbox=copy.deepcopy(self.task_def["inbox"]),
            processed=[],
            current_index=0,
            task_name=self.task_name,
            max_steps=len(self.task_def["inbox"]) * 3,
            current_steps=0
        )
        return self._get_obs()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        self._state.current_steps += 1
        done = False
        info = {}
        
        if self._state.current_index >= len(self._state.inbox):
            return self._get_obs(), Reward(value=0.0, reason="Inbox already empty."), True, info
            
        current_email = self._state.inbox[self._state.current_index]
        reward_val = 0.0
        reason = ""
        
        # Verify action is for current email
        if action.email_id != current_email["id"] and action.action_type != "skip":
            reward_val = -0.2
            reason = "Action ID does not match current email ID."
            return self._get_obs(), Reward(value=reward_val, reason=reason), done, info

        # Process actions
        if action.action_type == "classify_email":
            current_email["classification"] = action.classification
            reward_val = 0.5
            reason = f"Classified as {action.classification}."
            
        elif action.action_type == "reply_email":
            current_email["reply_text"] = action.reply_text
            reward_val = 0.3
            reason = "Replied to email."
            
        elif action.action_type == "mark_priority":
            current_email["priority_level"] = action.priority_level
            reward_val = 0.1
            reason = f"Marked priority as {action.priority_level}."
            
        elif action.action_type == "schedule_meeting":
            current_email["schedule_time"] = action.schedule_time
            reward_val = 0.2
            reason = f"Scheduled meeting for {action.schedule_time}."
            
        elif action.action_type == "skip":
            reason = "Skipped email."
        
        # In this task env, actions on current email accumulate. We advance to next email on 'skip' or if we want to force advance.
        # Let's simplify: after 3 actions on same email or 'skip'/'classify_email' down the line, we advance.
        # Actually, let's advance index if all necessary conditions for task are met, or if explicitly skipped.
        # To make it robust, we'll advance email index automatically if the main action is done for the task
        advance = False
        if self.task_name == "easy" and action.action_type == "classify_email":
            advance = True
        elif self.task_name == "medium" and action.action_type == "reply_email":
            advance = True
        elif self.task_name == "hard" and "schedule_time" in current_email and "reply_text" in current_email and "priority_level" in current_email:
            advance = True
        elif action.action_type == "skip":
            advance = True

        if advance:
            self._state.processed.append(current_email)
            self._state.current_index += 1
            
        if self._state.current_index >= len(self._state.inbox) or self._state.current_steps >= self._state.max_steps:
            done = True
            final_score = evaluate_task(self.task_name, self._state.processed)
            info["final_score"] = final_score
            reason += f" Task complete. Final Grader Score: {final_score:.2f}"
            
        return self._get_obs(), Reward(value=reward_val, reason=reason), done, info

    def state(self) -> State:
        return self._state
